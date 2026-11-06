// Naive row-major C(MxP) = A(MxN) * B(NxP), one CUDA thread per C element.
// Usage: matMul [M N P [double|float]] (defaults: 128 256 32 double)

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
        }                                                                  \
    } while (0)

constexpr unsigned BLOCK_DIM = 16;
constexpr size_t FULL_VALIDATION_WORK = 1ull << 26;

template <typename T>
__global__ void matMulKernel(const T* __restrict__ A, const T* __restrict__ B,
                             T* __restrict__ C, size_t M, size_t N, size_t P) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= P) return;

    T acc = T(0);
    const T* aRow = A + row * N;
    for (size_t k = 0; k < N; ++k) acc += aRow[k] * B[k * P + col];
    C[row * P + col] = acc;
}

template <typename T>
static T cpuDot(const T* A, const T* B, size_t row, size_t col, size_t N,
                size_t P) {
    T acc = T(0);
    for (size_t k = 0; k < N; ++k) acc += A[row * N + k] * B[k * P + col];
    return acc;
}

static std::vector<size_t> sampleIndices(size_t n) {
    const size_t candidates[] = {0, 1, n / 4, n / 2 - 1, n / 2,
                                 n / 2 + 1, 3 * n / 4,
                                 n >= 2 ? n - 2 : 0, n - 1};
    std::vector<size_t> indices;
    for (size_t i : candidates) {
        if (i < n) indices.push_back(i);
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

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
    *out = value;
    return true;
}

static double nowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

template <typename T>
static int runCase(size_t M, size_t N, size_t P, const char* typeName) {
    size_t aElems, bElems, cElems, work, aBytes, bBytes, cBytes;
    if (!checkedMul(M, N, &aElems) || !checkedMul(N, P, &bElems) ||
        !checkedMul(M, P, &cElems) || !checkedMul(aElems, P, &work) ||
        !checkedMul(aElems, sizeof(T), &aBytes) ||
        !checkedMul(bElems, sizeof(T), &bBytes) ||
        !checkedMul(cElems, sizeof(T), &cBytes)) {
        std::fprintf(stderr, "error: matrix size overflow\n");
        return EXIT_FAILURE;
    }
    if (ceilDiv(M, BLOCK_DIM) > 65535) {
        std::fprintf(stderr, "error: M exceeds CUDA grid.y limit\n");
        return EXIT_FAILURE;
    }

    // new T[n] does not zero-initialize float/double elements.
    std::unique_ptr<T[]> hA(new T[aElems]);
    std::unique_ptr<T[]> hB(new T[bElems]);
    std::unique_ptr<T[]> hC(new T[cElems]);
    T* A = hA.get();
    T* B = hB.get();
    T* C = hC.get();
    // Fixed varying patterns; denominator 16 is exact in float and double.
    for (size_t i = 0; i < aElems; ++i) A[i] = static_cast<T>(i % 17) / T(16);
    for (size_t i = 0; i < bElems; ++i) B[i] = static_cast<T>(i % 13) / T(16);

    T *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc((void**)&dA, aBytes));
    CHECK_CUDA(cudaMalloc((void**)&dB, bBytes));
    CHECK_CUDA(cudaMalloc((void**)&dC, cBytes));

    const dim3 block(BLOCK_DIM, BLOCK_DIM);
    const dim3 grid(ceilDiv(P, BLOCK_DIM), ceilDiv(M, BLOCK_DIM));

    CHECK_CUDA(cudaDeviceSynchronize());
    const double t0 = nowMs();
    CHECK_CUDA(cudaMemcpy(dA, A, aBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, bBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    const double t1 = nowMs();

    matMulKernel<T><<<grid, block>>>(dA, dB, dC, M, N, P);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    const double t2 = nowMs();

    CHECK_CUDA(cudaMemcpy(C, dC, cBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    const double t3 = nowMs();

    const bool fullValidation = work <= FULL_VALIDATION_WORK;
    const double tolerance = sizeof(T) == sizeof(double) ? 1e-9 : 1e-3;
    double maxAbs = 0.0, maxRel = 0.0;
    size_t checkedCount = 0;
    bool validationPassed = true;
    auto validate = [&](size_t row, size_t col) {
        const double ref = cpuDot(A, B, row, col, N, P);
        const double absErr = std::fabs(C[row * P + col] - ref);
        const double relErr = absErr / std::max(std::fabs(ref), 1e-30);
        maxAbs = std::max(maxAbs, absErr);
        maxRel = std::max(maxRel, relErr);
        if (absErr > tolerance + tolerance * std::fabs(ref)) validationPassed = false;
        ++checkedCount;
    };
    if (fullValidation) {
        for (size_t row = 0; row < M; ++row)
            for (size_t col = 0; col < P; ++col) validate(row, col);
    } else {
        const auto rows = sampleIndices(M);
        const auto cols = sampleIndices(P);
        for (size_t row : rows)
            for (size_t col : cols) validate(row, col);
    }

    const size_t blocksX = grid.x;
    const size_t blocksY = grid.y;
    const size_t totalThreads = blocksX * blocksY * BLOCK_DIM * BLOCK_DIM;
    const std::string validation =
        fullValidation ? "full" : "sampled:" + std::to_string(checkedCount);
    std::printf(
        "type,M,N,P,threads_per_block,blocks_x,blocks_y,total_threads,"
        "active_threads,H2D_ms,kernel_ms,D2H_ms,max_abs_error,max_rel_error,"
        "validation\n");
    std::printf("%s,%zu,%zu,%zu,%u,%zu,%zu,%zu,%zu,%.4f,%.4f,%.4f,"
                "%.6e,%.6e,%s\n",
                typeName, M, N, P, BLOCK_DIM * BLOCK_DIM, blocksX, blocksY,
                totalThreads, cElems, t1 - t0, t2 - t1, t3 - t2, maxAbs, maxRel,
                validation.c_str());

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    if (!validationPassed) {
        std::fprintf(stderr, "error: validation FAILED (max_abs=%.6e max_rel=%.6e)\n",
                     maxAbs, maxRel);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    if (argc != 1 && argc != 4 && argc != 5) {
        std::fprintf(stderr, "Usage: %s [M N P [double|float]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t M = 128, N = 256, P = 32;
    if (argc > 1 && (!parseDim(argv[1], &M) || !parseDim(argv[2], &N) ||
                     !parseDim(argv[3], &P))) {
        std::fprintf(stderr, "error: dimensions must be positive integers\n");
        return EXIT_FAILURE;
    }
    const std::string type = argc == 5 ? argv[4] : "double";
    if (type == "double") return runCase<double>(M, N, P, type.c_str());
    if (type == "float") return runCase<float>(M, N, P, type.c_str());
    std::fprintf(stderr, "error: invalid type '%s' (use double|float)\n", type.c_str());
    return EXIT_FAILURE;
}
