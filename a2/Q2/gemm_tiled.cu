// Q2 — Tiled Matrix Multiplication (CUDA, dynamic shared memory).
//
// Computes C(M x N) = A(M x K) * B(K x N), row-major, in double or float.
// For each case the program runs, in order:
//   1. a CPU reference gemm (timed once, used for correctness comparison),
//   2. the naive `gemm()` kernel (one thread per C element),
//   3. `tiled_gemm()` for a sweep of (tileX, tileY) configurations
//      (default: 8x8, 16x16, 32x32; override with extra "tileX tileY" args).
// Every kernel is warmed up once and then timed over GEMM_REPS launches
// (default 10, set GEMM_REPS=1 for a single un-warmed run, e.g. under ncu);
// the reported time is the median CUDA-event duration. The CPU reference is
// timed with std::chrono. All GPU results are compared element-by-element
// against the full CPU reference with a combined absolute+relative
// tolerance.
//
// Usage: gemm_tiled M K N [double|float] [tileX tileY]...
//   A is M x K, B is K x N, C is M x N. Out-of-bounds tile elements are
//   loaded as 0, so non-divisible M/K/N are safe.
//
// stdout: one '#' comment block per case (human readable, screenshot
// friendly) followed by a single CSV header (printed once) and one CSV row
// per timed invocation (cpu, gemm, one row per tiled config). All
// diagnostics go to stderr. Nonzero exit on bad args, CUDA errors, limit
// violations or validation failure.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",            \
                         cudaGetErrorString(err_), __FILE__, __LINE__);    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                 \
    } while (0)

// Default number of timed launches per kernel (median is reported).
// Override with the GEMM_REPS environment variable (positive integer;
// GEMM_REPS=1 skips the warm-up and times a single launch — useful for
// profiling with ncu).
constexpr int DEFAULT_REPS = 10;
constexpr size_t MAX_THREADS_PER_BLOCK = 1024;  // CUDA hardware limit
constexpr size_t MAX_GRID_Y = 65535;            // CUDA hardware limit

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Naive: one thread per C element, serial k-loop, all operands from global
// memory. C(M x N) = A(M x K) * B(K x N).
template <typename T>
__global__ void gemm(const T* __restrict__ A, const T* __restrict__ B,
                     T* __restrict__ C, size_t M, size_t K, size_t N) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    T acc = T(0);
    const T* aRow = A + row * K;
    for (size_t k = 0; k < K; ++k) acc += aRow[k] * B[k * N + col];
    C[row * N + col] = acc;
}

// Tiled: each block computes a tileY x tileX sub-tile of C using dynamic
// shared memory. The reduction is processed in k-chunks of tileX; in every
// stage the block cooperatively loads
//   As[tileY][tileX] <- A rows (blockIdx.y*tileY .. +tileY), cols k0..k0+tileX
//   Bs[tileX][tileX] <- B rows k0..k0+tileX,     cols blockIdx.x*tileX..+tileX
// with out-of-bounds elements set to 0 (no global read), so arbitrary M/K/N
// are safe. tileX/tileY are runtime parameters; the shared allocation is
// (tileY*tileX + tileX*tileX) * sizeof(T) bytes.
template <typename T>
__global__ void tiled_gemm(const T* __restrict__ A, const T* __restrict__ B,
                           T* __restrict__ C, size_t M, size_t K, size_t N,
                           size_t tileX, size_t tileY) {
    extern __shared__ unsigned char smemRaw[];
    T* As = reinterpret_cast<T*>(smemRaw);          // tileY x tileX
    T* Bs = As + tileY * tileX;                     // tileX x tileX

    const size_t row = blockIdx.y * tileY + threadIdx.y;
    const size_t col = blockIdx.x * tileX + threadIdx.x;

    T acc = T(0);
    for (size_t k0 = 0; k0 < K; k0 += tileX) {
        // Stage A tile: tileY rows x tileX cols (strided so any
        // blockDim <= tile dims still covers the whole tile).
        for (size_t r = threadIdx.y; r < tileY; r += blockDim.y) {
            const size_t aRow = blockIdx.y * tileY + r;
            for (size_t c = threadIdx.x; c < tileX; c += blockDim.x) {
                const size_t aCol = k0 + c;
                As[r * tileX + c] =
                    (aRow < M && aCol < K) ? A[aRow * K + aCol] : T(0);
            }
        }
        // Stage B tile: tileX rows x tileX cols.
        for (size_t r = threadIdx.y; r < tileX; r += blockDim.y) {
            const size_t bRow = k0 + r;
            for (size_t c = threadIdx.x; c < tileX; c += blockDim.x) {
                const size_t bCol = blockIdx.x * tileX + c;
                Bs[r * tileX + c] =
                    (bRow < K && bCol < N) ? B[bRow * N + bCol] : T(0);
            }
        }
        __syncthreads();

        // OOB elements were staged as 0, so a full tileX-wide inner loop is
        // exact even when K % tileX != 0.
        const T* aSub = As + threadIdx.y * tileX;
        for (size_t kk = 0; kk < tileX; ++kk)
            acc += aSub[kk] * Bs[kk * tileX + threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// CPU reference (ikj order: cache friendly, single-threaded, full C).
// ---------------------------------------------------------------------------
template <typename T>
static void cpu_gemm(const T* A, const T* B, T* C, size_t M, size_t K,
                     size_t N) {
    std::fill(C, C + M * N, T(0));
    for (size_t i = 0; i < M; ++i) {
        const T* aRow = A + i * K;
        T* cRow = C + i * N;
        for (size_t k = 0; k < K; ++k) {
            const T a = aRow[k];
            const T* bRow = B + k * N;
            for (size_t j = 0; j < N; ++j) cRow[j] += a * bRow[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static bool checkedMul(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static size_t ceilDiv(size_t a, size_t b) { return a / b + (a % b != 0); }

static bool parseDim(const char* s, size_t* out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0' || value == 0 || value > SIZE_MAX) return false;
    *out = static_cast<size_t>(value);
    return true;
}

static double nowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// GEMM_REPS override (see DEFAULT_REPS). Exits with an error on a missing,
// non-numeric, or non-positive value.
static int parseReps() {
    const char* s = std::getenv("GEMM_REPS");
    if (s == nullptr || *s == '\0') return DEFAULT_REPS;
    char* end = nullptr;
    const long value = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || value <= 0) {
        std::fprintf(stderr,
                     "error: GEMM_REPS must be a positive integer, got '%s'\n",
                     s);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

// Times `launch` over `reps` CUDA-event-measured launches (one warm-up
// launch first when reps > 1) and returns the median duration in
// milliseconds. Every launch is checked with cudaGetLastError and every
// event pair is synchronized, so a failing kernel aborts the program.
template <typename LaunchFn>
static float timeKernelMedian(LaunchFn&& launch, int reps) {
    if (reps > 1) {
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    std::vector<float> times;
    times.reserve(static_cast<size_t>(reps));
    for (int i = 0; i < reps; ++i) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    return times[static_cast<size_t>(reps) / 2];
}

static const char* CSV_HEADER =
    "type,M,K,N,kernel,tile_x,tile_y,threads_per_block,blocks_x,blocks_y,"
    "total_threads,active_threads,smem_per_block_B,time_ms,max_abs_error,"
    "max_rel_error,validation";

static void printCsvRow(const char* typeName, size_t M, size_t K, size_t N,
                        const char* kernel, size_t tileX, size_t tileY,
                        size_t threadsPerBlock, size_t blocksX, size_t blocksY,
                        size_t activeThreads, size_t smemBytes, double ms,
                        double maxAbs, double maxRel,
                        const std::string& validation) {
    const size_t totalThreads = threadsPerBlock * blocksX * blocksY;
    std::printf("%s,%zu,%zu,%zu,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%.4f,"
                "%.6e,%.6e,%s\n",
                typeName, M, K, N, kernel, tileX, tileY, threadsPerBlock,
                blocksX, blocksY, totalThreads, activeThreads, smemBytes, ms,
                maxAbs, maxRel, validation.c_str());
}

// ---------------------------------------------------------------------------
// Case driver
// ---------------------------------------------------------------------------
struct TileConfig {
    size_t x, y;  // tileX (columns of C / k-chunk), tileY (rows of C)
};

template <typename T>
static int runCase(size_t M, size_t K, size_t N, const char* typeName,
                   const std::vector<TileConfig>& tiles) {
    size_t aElems, bElems, cElems, aBytes, bBytes, cBytes;
    if (!checkedMul(M, K, &aElems) || !checkedMul(K, N, &bElems) ||
        !checkedMul(M, N, &cElems) ||
        !checkedMul(aElems, sizeof(T), &aBytes) ||
        !checkedMul(bElems, sizeof(T), &bBytes) ||
        !checkedMul(cElems, sizeof(T), &cBytes)) {
        std::fprintf(stderr, "error: matrix size overflow\n");
        return EXIT_FAILURE;
    }

    // Device sanity: shared memory ceiling and block-thread limit.
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    const size_t maxSmem = static_cast<size_t>(prop.sharedMemPerBlock);
    if (ceilDiv(M, 16) > MAX_GRID_Y) {
        std::fprintf(stderr, "error: naive 16x16 blocks give grid.y=%zu "
                             "(limit %zu) for M=%zu\n",
                     ceilDiv(M, 16), MAX_GRID_Y, M);
        return EXIT_FAILURE;
    }
    for (const TileConfig& t : tiles) {
        size_t smem = 0;
        if (!checkedMul(t.x * t.y + t.x * t.x, sizeof(T), &smem) ||
            smem > maxSmem) {
            std::fprintf(stderr,
                         "error: tile %zux%zu needs %zu bytes of shared "
                         "memory, device limit is %zu\n",
                         t.x, t.y, smem, maxSmem);
            return EXIT_FAILURE;
        }
        if (t.x * t.y > MAX_THREADS_PER_BLOCK) {
            std::fprintf(stderr, "error: tile %zux%zu needs %zu threads per "
                                 "block (limit %zu)\n",
                         t.x, t.y, t.x * t.y, MAX_THREADS_PER_BLOCK);
            return EXIT_FAILURE;
        }
        if (ceilDiv(M, t.y) > MAX_GRID_Y) {
            std::fprintf(stderr, "error: tileY=%zu gives grid.y=%zu (limit "
                                 "%zu) for M=%zu\n",
                         t.y, ceilDiv(M, t.y), MAX_GRID_Y, M);
            return EXIT_FAILURE;
        }
    }

    // Fixed varying patterns; denominator 16 is exact in float and double.
    std::unique_ptr<T[]> hA(new T[aElems]);
    std::unique_ptr<T[]> hB(new T[bElems]);
    std::unique_ptr<T[]> hRef(new T[cElems]);
    std::unique_ptr<T[]> hC(new T[cElems]);
    T* A = hA.get();
    T* B = hB.get();
    T* C = hC.get();
    for (size_t i = 0; i < aElems; ++i) A[i] = static_cast<T>(i % 17) / T(16);
    for (size_t i = 0; i < bElems; ++i) B[i] = static_cast<T>(i % 13) / T(16);

    T *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc((void**)&dA, aBytes));
    CHECK_CUDA(cudaMalloc((void**)&dB, bBytes));
    CHECK_CUDA(cudaMalloc((void**)&dC, cBytes));
    CHECK_CUDA(cudaMemcpy(dA, A, aBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, bBytes, cudaMemcpyHostToDevice));

    // CPU reference: computed once per case, timed, reused for all
    // kernel validations (no redundant re-computation or extra copies).
    CHECK_CUDA(cudaDeviceSynchronize());
    const double c0 = nowMs();
    cpu_gemm(A, B, hRef.get(), M, K, N);
    const double cpuMs = nowMs() - c0;
    const int reps = parseReps();

    std::printf("# case: M=%zu K=%zu N=%zu type=%s | A=%zux%zu B=%zux%zu "
                "C=%zux%zu\n",
                M, K, N, typeName, M, K, K, N, M, N);
    char timingMode[64];
    if (reps == 1)
        std::snprintf(timingMode, sizeof(timingMode), "1 run (no warm-up)");
    else
        std::snprintf(timingMode, sizeof(timingMode), "%d runs + warm-up",
                      reps);
    std::printf("# device: %s (sm_%d%d), cpu reference: %.2f ms, kernel "
                "timing: %s, median CUDA-event ms\n",
                prop.name, prop.major, prop.minor, cpuMs, timingMode);

    const double tolerance = sizeof(T) == sizeof(double) ? 1e-9 : 1e-3;
    const size_t blocksXNaive = ceilDiv(N, 16);
    const size_t blocksYNaive = ceilDiv(M, 16);
    bool allPassed = true;

    // Validate one GPU result buffer against the full CPU reference
    // (every C element; O(M*N) per kernel).
    auto validate = [&](double& maxAbs, double& maxRel) {
        maxAbs = 0.0;
        maxRel = 0.0;
        for (size_t r = 0; r < M; ++r) {
            for (size_t c = 0; c < N; ++c) {
                const double ref = static_cast<double>(hRef[r * N + c]);
                const double got = static_cast<double>(C[r * N + c]);
                const double absErr = std::fabs(got - ref);
                const double relErr = absErr / std::max(std::fabs(ref), 1e-30);
                maxAbs = std::max(maxAbs, absErr);
                maxRel = std::max(maxRel, relErr);
                if (absErr > tolerance + tolerance * std::fabs(ref))
                    allPassed = false;
            }
        }
    };

    // CPU reference row in the CSV (validation of GPU results is always
    // full, so the case tag is "full").
    printCsvRow(typeName, M, K, N, "cpu", 0, 0, 0, 0, 0, 0, 0, cpuMs, 0.0,
                0.0, "full");

    // --- naive gemm() ---
    {
        const dim3 block(16, 16);
        const dim3 grid(blocksXNaive, blocksYNaive);
        const float ms = timeKernelMedian(
            [&] { gemm<T><<<grid, block>>>(dA, dB, dC, M, K, N); }, reps);
        CHECK_CUDA(cudaMemcpy(C, dC, cBytes, cudaMemcpyDeviceToHost));

        double maxAbs, maxRel;
        validate(maxAbs, maxRel);
        const double speedup = cpuMs / ms;
        std::printf("# gemm (naive 16x16): %s  kernel=%.4f ms  speedup=%.1fx"
                    "  max_rel=%.3e\n",
                    allPassed ? "PASS" : "FAIL", ms, speedup, maxRel);
        printCsvRow(typeName, M, K, N, "gemm", 0, 0, 256, blocksXNaive,
                    blocksYNaive, cElems, 0, ms, maxAbs, maxRel, "full");
    }

    // --- tiled_gemm() sweep ---
    for (const TileConfig& t : tiles) {
        const dim3 block(t.x, t.y);
        const dim3 grid(ceilDiv(N, t.x), ceilDiv(M, t.y));
        const size_t smem = (t.y * t.x + t.x * t.x) * sizeof(T);
        const float ms = timeKernelMedian(
            [&] {
                tiled_gemm<T><<<grid, block, smem>>>(dA, dB, dC, M, K, N, t.x,
                                                     t.y);
            },
            reps);
        CHECK_CUDA(cudaMemcpy(C, dC, cBytes, cudaMemcpyDeviceToHost));

        double maxAbs, maxRel;
        validate(maxAbs, maxRel);
        const double speedup = cpuMs / ms;
        std::printf("# tiled %2zux%-2zu (smem=%zu B): %s  kernel=%.4f ms"
                    "  speedup=%.1fx  max_rel=%.3e\n",
                    t.x, t.y, smem, allPassed ? "PASS" : "FAIL", ms, speedup,
                    maxRel);
        printCsvRow(typeName, M, K, N, "tiled", t.x, t.y, t.x * t.y, grid.x,
                    grid.y, cElems, smem, ms, maxAbs, maxRel, "full");
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    if (!allPassed) {
        std::fprintf(stderr, "error: validation FAILED for M=%zu K=%zu N=%zu\n",
                     M, K, N);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    // Optional trailing "tileX tileY" pairs override the default sweep.
    if (argc < 4) {
        std::fprintf(stderr,
                     "Usage: %s M K N [double|float] [tileX tileY]...\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    size_t M, K, N;
    if (!parseDim(argv[1], &M) || !parseDim(argv[2], &K) ||
        !parseDim(argv[3], &N)) {
        std::fprintf(stderr, "error: dimensions must be positive integers\n");
        return EXIT_FAILURE;
    }

    int argi = 4;
    std::string type = "double";
    if (argi < argc &&
        (std::strcmp(argv[argi], "double") == 0 ||
         std::strcmp(argv[argi], "float") == 0)) {
        type = argv[argi];
        ++argi;
    }

    std::vector<TileConfig> tiles;
    if (argi == argc) {
        tiles = {{8, 8}, {16, 16}, {32, 32}};
    } else {
        if (((argc - argi) % 2) != 0) {
            std::fprintf(stderr,
                         "error: tile sizes must come in tileX tileY pairs\n");
            return EXIT_FAILURE;
        }
        for (; argi + 1 < argc; argi += 2) {
            TileConfig t;
            if (!parseDim(argv[argi], &t.x) || !parseDim(argv[argi + 1], &t.y)) {
                std::fprintf(stderr,
                             "error: tile sizes must be positive integers\n");
                return EXIT_FAILURE;
            }
            tiles.push_back(t);
        }
    }

    std::puts(CSV_HEADER);
    if (type == "double")
        return runCase<double>(M, K, N, type.c_str(), tiles);
    return runCase<float>(M, K, N, type.c_str(), tiles);
}
