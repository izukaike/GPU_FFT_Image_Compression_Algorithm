
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include <cufft.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
//using namespace std;
static inline void checkCuda(const char* what, cudaError_t e) {
    if (e == cudaSuccess) return;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
}


struct fft_obj {
    char     magic[4] = {'F','F','T','C'}; 
    uint32_t version  = 1;                 
    uint32_t width    = 0;
    uint32_t height   = 0;
    int32_t  channels = 3;               
    uint32_t bytes    = 0;                 
    uint32_t reserved = 0;

    void print() const {
        std::string m(magic, magic+4);
        std::printf(
            "fft_obj:\n"
            "  magic     = \"%s\"\n"
            "  version   = %u\n"
            "  size      = %u x %u\n"
            "  channels  = %d\n"
            "  bytes     = %u\n"
            "  reserved  = %u\n",
            m.c_str(), version, width, height, channels, bytes, reserved
        );
    }
};


struct FftTilesOut {
    float*        d_tiles_real  = nullptr; 
    cufftComplex* d_tiles_freq  = nullptr; 
    int           tiles  = 0;              
    int           batch  = 0;              
    int           colsC  = 0;              
    cufftHandle   planR2C = 0;             
};

// ------------------------------------------------------------
// Compute top-left origin for every tile (CPU).
// Stride S < B -> overlap. We “snap” last tiles to the border.
// ------------------------------------------------------------
static void make_tile_origins(int W, int H, int B, int S,
                              std::vector<int>& x0, std::vector<int>& y0)
{
    x0.clear(); y0.clear();

    for (int y = 0; ; ) {
        for (int x = 0; ; ) {
            x0.push_back(x);
            y0.push_back(y);

            if (x + B >= W) break;      // last tile in this row
            x += S;
            if (x + B > W) x = W - B;   // snap to right edge
        }
        if (y + B >= H) break;          // last row of tiles
        y += S;
        if (y + B > H) y = H - B;       // snap to bottom edge
    }
}

// ------------------------------------------------------------
// KERNEL: build a BxB 2D Hann window (separable)
// ------------------------------------------------------------
__global__ void build_hann2d(float* W2d, int B)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 0..B-1
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 0..B-1
    if (x >= B || y >= B) return;

    const float two_pi = 6.283185307179586f;
    float wx = 0.5f * (1.f - __cosf(two_pi * x / (B - 1)));
    float wy = 0.5f * (1.f - __cosf(two_pi * y / (B - 1)));
    W2d[y*B + x] = wx * wy;
}

// ------------------------------------------------------------
// KERNEL: gather RGB tiles into a single batched real buffer
// (and apply Hann window). Batch = tile*3 + channel.
// Layout: out_real[batch][B][B]
// ------------------------------------------------------------
__global__ void gather_window_tiles_rgb_to_batch(
    const unsigned char* d_rgb, int W, int H,
    const int* d_x0, const int* d_y0, int tiles,
    const float* d_W2d, int B,
    float* out_real)
{
    int b = blockIdx.z;         // which (tile,channel)
    if (b >= tiles * 3) return;

    int t = b / 3;              // tile index
    int c = b % 3;              // channel (0,1,2)

    int tx = blockIdx.x * blockDim.x + threadIdx.x; // 0..B-1
    int ty = blockIdx.y * blockDim.y + threadIdx.y; // 0..B-1
    if (tx >= B || ty >= B) return;

    const int X0 = d_x0[t], Y0 = d_y0[t];
    int x = X0 + tx, y = Y0 + ty;

    // clamp to image (safe at borders)
    x = (x < 0) ? 0 : (x >= W ? W-1 : x);
    y = (y < 0) ? 0 : (y >= H ? H-1 : y);

    const int i3 = (y*W + x) * 3;
    const float pix = d_rgb[i3 + c] * (1.0f/255.0f);
    const float w   = d_W2d[ty*B + tx];

    const size_t tileSize = size_t(B) * B;
    out_real[size_t(b)*tileSize + ty*B + tx] = pix * w;
}

// ------------------------------------------------------------
// Build tiles, window them, and run batched cuFFT (R2C).
// Writes out.d_tiles_real and out.d_tiles_freq.
// ------------------------------------------------------------
static bool window_and_fft_rgb_gpu(const unsigned char* d_rgb, int W, int H,
                                   int B, int S,
                                   FftTilesOut& out)
{
    // ---- 1) figure out where each tile starts (CPU) ----
    std::vector<int> h_x0, h_y0;
    make_tile_origins(W, H, B, S, h_x0, h_y0);
    const int tiles = (int)h_x0.size();
    const int batch = tiles * 3;
    const int colsC = B/2 + 1;

    std::printf("Tiles: %d (B=%d, S=%d) -> batch=%d, colsC=%d\n",
                tiles, B, S, batch, colsC);

    // ---- 2) upload tile origins + make Hann window on GPU ----
    int *d_x0=nullptr, *d_y0=nullptr;
    checkCuda("cudaMalloc d_x0", cudaMalloc(&d_x0, tiles*sizeof(int)));
    checkCuda("cudaMalloc d_y0", cudaMalloc(&d_y0, tiles*sizeof(int)));
    checkCuda("cudaMemcpy x0",   cudaMemcpy(d_x0, h_x0.data(), tiles*sizeof(int), cudaMemcpyHostToDevice));
    checkCuda("cudaMemcpy y0",   cudaMemcpy(d_y0, h_y0.data(), tiles*sizeof(int), cudaMemcpyHostToDevice));

    float* d_W2d=nullptr;
    checkCuda("cudaMalloc W2d", cudaMalloc(&d_W2d, B*B*sizeof(float)));
    {
        dim3 bs(16,16), gs((B+15)/16, (B+15)/16);
        build_hann2d<<<gs, bs>>>(d_W2d, B);
        checkCuda("build_hann2d launch", cudaGetLastError());
    }

    // ---- 3) big real buffer for all windowed tiles ----
    float* d_tiles_real=nullptr;
    const size_t realPerTile = size_t(B)*B;
    checkCuda("cudaMalloc tiles_real",
              cudaMalloc(&d_tiles_real, size_t(batch)*realPerTile*sizeof(float)));

    // ---- 4) gather pixels -> apply window -> write to batch ----
    {
        dim3 bs(16,16);
        dim3 gs((B+bs.x-1)/bs.x, (B+bs.y-1)/bs.y, batch);
        gather_window_tiles_rgb_to_batch<<<gs, bs>>>(
            d_rgb, W, H, d_x0, d_y0, tiles, d_W2d, B, d_tiles_real);
        checkCuda("gather launch", cudaGetLastError());
    }

    // ---- 5) batched cuFFT R2C ----
    cufftHandle planR2C = 0;
    const int n[2]       = { B, B };
    const int inembed[2] = { B, B };
    const int onembed[2] = { B, colsC };
    const int istride=1,  ostride=1;
    const int idist=B*B,  odist=B*colsC;

    if (cufftPlanMany(&planR2C, 2, (int*)n,
                      (int*)inembed,  istride, idist,
                      (int*)onembed,  ostride, odist,
                      CUFFT_R2C, batch) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftPlanMany R2C failed\n");
        cudaFree(d_x0); cudaFree(d_y0); cudaFree(d_W2d);
        cudaFree(d_tiles_real);
        return false;
    }

    cufftComplex* d_tiles_freq=nullptr;
    checkCuda("cudaMalloc tiles_freq",
              cudaMalloc(&d_tiles_freq, size_t(batch)*B*colsC*sizeof(cufftComplex)));

    if (cufftExecR2C(planR2C,
                     (cufftReal*)d_tiles_real,
                     d_tiles_freq) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftExecR2C failed\n");
        cufftDestroy(planR2C);
        cudaFree(d_tiles_freq);
        cudaFree(d_x0); cudaFree(d_y0); cudaFree(d_W2d);
        cudaFree(d_tiles_real);
        return false;
    }

    // ---- 6) hand back results ----
    out.d_tiles_real = d_tiles_real;
    out.d_tiles_freq = d_tiles_freq;
    out.tiles  = tiles;
    out.batch  = batch;
    out.colsC  = colsC;
    out.planR2C = planR2C;

    cudaFree(d_x0); cudaFree(d_y0); cudaFree(d_W2d);
    return true;
}

// ------------------------------------------------------------
// Pack complex frequency slabs as a simple float array:
// Header floats: [B, S, colsC, keepK=0]. Then Re,Im interleaved.
// ------------------------------------------------------------
static bool pack_dense_fft_as_floats(const FftTilesOut& tiles, int B, int S,
                                     std::vector<float>& out)
{
    const int colsC = tiles.colsC;
    const size_t N  = size_t(tiles.batch) * B * colsC;

    std::vector<cufftComplex> h(N);
    cudaError_t ce = cudaMemcpy(h.data(), tiles.d_tiles_freq,
                                N*sizeof(cufftComplex),
                                cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "pack_dense_fft_as_floats: %s\n", cudaGetErrorString(ce));
        return false;
    }

    out.clear();
    out.reserve(4 + 2*N);

    // tiny header
    out.push_back(float(B));
    out.push_back(float(S));
    out.push_back(float(colsC));
    out.push_back(0.0f); // keepK=0 means “dense” (no sparsity)

    // Re/Im pairs
    for (size_t i = 0; i < N; ++i) {
        out.push_back(h[i].x);
        out.push_back(h[i].y);
    }
    return true;
}

// ------------------------------------------------------------
// Write file: [fft_obj header][payload bytes]
// ------------------------------------------------------------
static bool bin_out(const char* filename,
                    const fft_obj& hdr_in,
                    const void* payload,
                    size_t nbytes)
{
    fft_obj hdr = hdr_in;
    std::memcpy(hdr.magic, "FFTC", 4);
    hdr.bytes = (uint32_t)nbytes;

    std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
    if (!ofs) { std::fprintf(stderr, "bin_out: open %s failed\n", filename); return false; }

    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!ofs) { std::fprintf(stderr, "bin_out: header write failed\n"); return false; }

    if (nbytes && payload) {
        ofs.write(reinterpret_cast<const char*>(payload), (std::streamsize)nbytes);
        if (!ofs) { std::fprintf(stderr, "bin_out: payload write failed\n"); return false; }
    }
    return true;
}

// ------------------------------------------------------------
// Read whole file back into memory
// ------------------------------------------------------------
static bool bin_read_all(const char* filename,
                         fft_obj& hdr_out,
                         std::vector<uint8_t>& payload_out)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) { std::fprintf(stderr, "bin_read_all: open %s failed\n", filename); return false; }

    ifs.read(reinterpret_cast<char*>(&hdr_out), sizeof(hdr_out));
    if (!ifs) { std::fprintf(stderr, "bin_read_all: header read failed\n"); return false; }

    if (std::memcmp(hdr_out.magic, "FFTC", 4) != 0) {
        std::fprintf(stderr, "bin_read_all: bad magic (not FFTC)\n"); return false;
    }

    payload_out.resize(hdr_out.bytes);
    if (hdr_out.bytes) {
        ifs.read(reinterpret_cast<char*>(payload_out.data()), (std::streamsize)hdr_out.bytes);
        if (!ifs) { std::fprintf(stderr, "bin_read_all: payload read failed\n"); return false; }
    }
    return true;
}

// ------------------------------------------------------------
// KERNEL: overlap-add tiles to reconstruct image.
// We divide by (B*B) (cuFFT C2R is unnormalized) and keep a per-
// pixel hit count for simple normalization at overlaps.
// ------------------------------------------------------------
__global__ void scatter_tiles_to_image_rgb(
    const float* d_tiles_real, int B,
    const int* d_x0, const int* d_y0, int tiles,
    int W, int H, float* d_img_accum, float* d_wsum)
{
    int b = blockIdx.z; if (b >= tiles*3) return;
    int t = b / 3; int c = b % 3;

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= B || ty >= B) return;

    const int X0 = d_x0[t], Y0 = d_y0[t];
    int x = X0 + tx, y = Y0 + ty;
    if (x < 0 || x >= W || y < 0 || y >= H) return;

    const size_t tileSize = size_t(B) * B;
    const float v = d_tiles_real[size_t(b)*tileSize + ty*B + tx] * (1.0f / (B*B));

    const size_t i = (size_t)y*W + x;
    atomicAdd(&d_img_accum[3*i + c], v);
    atomicAdd(&d_wsum[i], 1.0f);
}

// ------------------------------------------------------------
// KERNEL: divide by weight and pack back to 8-bit RGB
// ------------------------------------------------------------
__global__ void normalize_and_pack_uchar(
    const float* d_img_accum, const float* d_wsum,
    int W, int H, unsigned char* d_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const size_t i = (size_t)y*W + x;
    const float w = d_wsum[i];

    float r=0.f, g=0.f, b=0.f;
    if (w > 0.f) {
        r = d_img_accum[3*i+0] / w;
        g = d_img_accum[3*i+1] / w;
        b = d_img_accum[3*i+2] / w;
    }

    // tiny helper to clamp to [0,255]
    auto to_u8 = [](float v)->unsigned char {
        v = fminf(fmaxf(v, 0.f), 1.f);
        return (unsigned char)lrintf(v * 255.f);
    };

    // mild gain so the preview isn’t too dark
    const float gain = 16.0f;
    r *= gain; g *= gain; b *= gain;

    d_out[3*i+0] = to_u8(r);
    d_out[3*i+1] = to_u8(g);
    d_out[3*i+2] = to_u8(b);
}

// ------------------------------------------------------------
// Decompress dense float blob -> reconstruct image (GPU)
// Blob: [B, S, colsC, 0,  Re0,Im0, Re1,Im1, ... ]
// ------------------------------------------------------------
static bool decompress_dense_fft_blob_to_image_gpu(
    const std::vector<uint8_t>& blob, int W, int H, unsigned char* h_rgb_out)
{
    if (blob.size() < 4*sizeof(float)) {
        std::fprintf(stderr, "decompress: blob too small\n");
        return false;
    }
    const float* pf = reinterpret_cast<const float*>(blob.data());
    const size_t nf = blob.size() / sizeof(float);

    const int B     = int(pf[0] + 0.5f);
    const int S     = int(pf[1] + 0.5f);
    const int colsC = int(pf[2] + 0.5f);
    const int keepK = int(pf[3] + 0.5f);
    if (keepK != 0) {
        std::fprintf(stderr, "decompress: expected dense keepK=0\n");
        return false;
    }

    // rebuild tile origins to scatter tiles back
    std::vector<int> h_x0, h_y0;
    make_tile_origins(W, H, B, S, h_x0, h_y0);
    const int tiles = (int)h_x0.size();
    const int batch = tiles * 3;
    const int binsPer = B * colsC;
    const size_t N = (size_t)batch * binsPer;

    if (nf != 4 + 2*N) {
        std::fprintf(stderr, "decompress: count mismatch (got %zu, need %zu)\n",
                     nf, (size_t)(4 + 2*N));
        return false;
    }

    // host complex frequency buffer
    std::vector<cufftComplex> h_freq(N);
    for (size_t i = 0; i < N; ++i) {
        h_freq[i].x = pf[4 + 2*i + 0];
        h_freq[i].y = pf[4 + 2*i + 1];
    }

    // device: freq -> real (C2R)
    cufftComplex* d_freq=nullptr; float* d_real=nullptr;
    checkCuda("malloc d_freq", cudaMalloc(&d_freq, N*sizeof(cufftComplex)));
    checkCuda("malloc d_real", cudaMalloc(&d_real, (size_t)batch*B*B*sizeof(float)));
    checkCuda("cpy freq", cudaMemcpy(d_freq, h_freq.data(), N*sizeof(cufftComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    const int n[2] = { B, B };
    const int onembed[2] = { B, colsC }; // freq layout
    const int inembed[2] = { B, B };     // real layout

    if (cufftPlanMany(&plan, 2, (int*)n,
                      (int*)onembed, 1, B*colsC,
                      (int*)inembed, 1, B*B,
                      CUFFT_C2R, batch) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftPlanMany C2R failed\n");
        cudaFree(d_freq); cudaFree(d_real);
        return false;
    }
    if (cufftExecC2R(plan, d_freq, (cufftReal*)d_real) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftExecC2R failed\n");
        cufftDestroy(plan);
        cudaFree(d_freq); cudaFree(d_real);
        return false;
    }

    // upload tile origins
    int *d_x0=nullptr, *d_y0=nullptr;
    checkCuda("malloc d_x0", cudaMalloc(&d_x0, tiles*sizeof(int)));
    checkCuda("malloc d_y0", cudaMalloc(&d_y0, tiles*sizeof(int)));
    checkCuda("cpy x0", cudaMemcpy(d_x0, h_x0.data(), tiles*sizeof(int), cudaMemcpyHostToDevice));
    checkCuda("cpy y0", cudaMemcpy(d_y0, h_y0.data(), tiles*sizeof(int), cudaMemcpyHostToDevice));

    // accumulate tiles into image, then normalize and pack
    float *d_accum=nullptr, *d_wsum=nullptr; unsigned char* d_out=nullptr;
    checkCuda("malloc accum", cudaMalloc(&d_accum, (size_t)W*H*3*sizeof(float)));
    checkCuda("malloc wsum",  cudaMalloc(&d_wsum,  (size_t)W*H*sizeof(float)));
    checkCuda("malloc out",   cudaMalloc(&d_out,   (size_t)W*H*3*sizeof(unsigned char)));
    checkCuda("zero accum",   cudaMemset(d_accum, 0, (size_t)W*H*3*sizeof(float)));
    checkCuda("zero wsum",    cudaMemset(d_wsum,  0, (size_t)W*H*sizeof(float)));

    {
        dim3 bs(16,16), gs((B+15)/16, (B+15)/16, batch);
        scatter_tiles_to_image_rgb<<<gs, bs>>>(d_real, B, d_x0, d_y0, tiles, W, H, d_accum, d_wsum);
        checkCuda("scatter launch", cudaGetLastError());
    }
    {
        dim3 bs(16,16), gs((W+15)/16, (H+15)/16);
        normalize_and_pack_uchar<<<gs, bs>>>(d_accum, d_wsum, W, H, d_out);
        checkCuda("normalize launch", cudaGetLastError());
    }

    // copy preview image back to host
    checkCuda("cpy out d->h", cudaMemcpy(h_rgb_out, d_out, (size_t)W*H*3, cudaMemcpyDeviceToHost));

    // cleanup
    cufftDestroy(plan);
    cudaFree(d_freq); cudaFree(d_real);
    cudaFree(d_x0);   cudaFree(d_y0);
    cudaFree(d_accum); cudaFree(d_wsum); cudaFree(d_out);
    return true;
}

// ------------------------------------------------------------
// MAIN
// Usage: ./main <in.png> <out.fftc>
// (This also writes a PNG preview next to the output name.)
// ------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <in.png> <out.fftc>\n", argv[0]);
        return 1;
    }

    // If you prefer hardcoded paths (as in your original), replace argv.
    const char* in_png   = "/home/ikedionwui/code/mpplabs/Final_Project/apple.png";
    const char* out_fftc = "/home/ikedionwui/code/mpplabs/Final_Project/apple_fft.png";

    // ---- Load PNG (force 3 channels: RGB) ----
    int W=0, H=0, comp_in=0;
    unsigned char* img = stbi_load(in_png, &W, &H, &comp_in, 3);
    if (!img) {
        std::fprintf(stderr, "Failed to load %s\n", in_png);
        if (const char* why = stbi_failure_reason()) std::fprintf(stderr, "stb: %s\n", why);
        return 2;
    }
    const size_t nbytes = (size_t)W * H * 3;
    std::printf("Loaded %s (%dx%d, forced RGB)\n", in_png, W, H);

    // ---- Device copy of input RGB ----
    unsigned char* d_in=nullptr;
    checkCuda("cudaMalloc d_in", cudaMalloc(&d_in, nbytes));
    checkCuda("cudaMemcpy h->d", cudaMemcpy(d_in, img, nbytes, cudaMemcpyHostToDevice));

    // ---- Make tiles + window + batched FFT on GPU ----
    const int B = 32;   // tile size (try 16/32/64)
    const int S = 16;   // stride (S<B => overlap)
    FftTilesOut tiles;

    if (!window_and_fft_rgb_gpu(d_in, W, H, B, S, tiles)) {
        std::fprintf(stderr, "window_and_fft_rgb_gpu failed\n");
        stbi_image_free(img);
        cudaFree(d_in);
        return 3;
    }

    // ---- Pack frequency slabs to a dense float blob ----
    std::vector<float> fblob;
    if (!pack_dense_fft_as_floats(tiles, B, S, fblob)) {
        std::fprintf(stderr, "pack_dense_fft_as_floats failed\n");
        stbi_image_free(img);
        cudaFree(d_in);
        cufftDestroy(tiles.planR2C);
        cudaFree(tiles.d_tiles_freq);
        cudaFree(tiles.d_tiles_real);
        return 4;
    }

    // ---- Write .fftc file ----
    fft_obj hdr{};
    std::memcpy(hdr.magic, "FFTC", 4);
    hdr.version  = 1;
    hdr.width    = (uint32_t)W;
    hdr.height   = (uint32_t)H;
    hdr.channels = 3;
    hdr.bytes    = (uint32_t)(fblob.size() * sizeof(float));
    hdr.reserved = 0;

    if (!bin_out(out_fftc, hdr, fblob.data(), fblob.size()*sizeof(float))) {
        std::fprintf(stderr, "bin_out failed\n");
        stbi_image_free(img);
        cudaFree(d_in);
        cufftDestroy(tiles.planR2C);
        cudaFree(tiles.d_tiles_freq);
        cudaFree(tiles.d_tiles_real);
        return 5;
    }
    std::printf("Wrote %s (payload %.2f MB)\n", out_fftc, hdr.bytes/1048576.0);

    // ---- Optional: read back + decompress to a quick PNG preview ----
    std::vector<uint8_t> payload;
    fft_obj hdr_in{};
    if (!bin_read_all(out_fftc, hdr_in, payload)) {
        std::fprintf(stderr, "bin_read_all failed (skip preview)\n");
    } else {
        const int dW = (int)hdr_in.width;
        const int dH = (int)hdr_in.height;
        std::vector<unsigned char> preview((size_t)dW * dH * 3);

        if (!decompress_dense_fft_blob_to_image_gpu(payload, dW, dH, preview.data())) {
            std::fprintf(stderr, "decompress failed (skip preview)\n");
        } else {
            // Write a PNG preview next to the .fftc (same name + ".preview.png")
            std::string preview_path = std::string(out_fftc);
            if (!stbi_write_png(preview_path.c_str(), dW, dH, 3, preview.data(), dW*3)) {
                std::fprintf(stderr, "stbi_write_png failed\n");
            } else {
                std::printf("Saved preview: %s\n", preview_path.c_str());
            }
        }
    }

    // ---- Cleanup ----
    stbi_image_free(img);
    cudaFree(d_in);
    cufftDestroy(tiles.planR2C);
    cudaFree(tiles.d_tiles_freq);
    cudaFree(tiles.d_tiles_real);

    return 0;
}
