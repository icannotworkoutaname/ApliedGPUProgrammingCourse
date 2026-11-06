// 
// benchmark a native QFT on QuEST CUDA backend
// Usage:  ./qft_bench [numQubits=24] [numReps=5]
//

#include "quest.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double nowSec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

// QFT: Hadamard gate + controlled phases per qubit
// then reverse the order pi/2, pi/4, pi/8 ....
static void QFT(Qureg q, int n, long *numGates) {
    for (int t = n - 1; t >= 0; t--) {
        applyHadamard(q, t);
        (*numGates)++;
        for (int c = t - 1; c >= 0; c--) {
            applyTwoQubitPhaseShift(q, c, t, M_PI / (1 << (t - c)));
            (*numGates)++;
        }
    }
    for (int i = 0; i < n / 2; i++) {
        applySwap(q, i, n - 1 - i);
        (*numGates)++;
    }
}

int main(int argc, char **argv) {
    int n    = (argc > 1) ? atoi(argv[1]) : 24;
    int reps = (argc > 2) ? atoi(argv[2]) : 5;

    initQuESTEnv();
    Qureg q = createForcedQureg(n);
    printf("QFT benchmark: %d qubits, %d reps, %s backend, %.2f GiB state vector\n",
           n, reps, q.isGpuAccelerated ? "GPU" : "CPU",
           (double)(1LL << n) * sizeof(qcomp) / (1 << 30));

    // Warm-up: high cost CUDA context
    long gates = 0;
    initPlusState(q);
    QFT(q, n, &gates);
    syncQuESTEnv();

    double best = 1e30, total = 0;
    for (int r = 0; r < reps; r++) {
        initPlusState(q);
        syncQuESTEnv();                  

        gates = 0;
        double t0 = nowSec();
        QFT(q, n, &gates);
        syncQuESTEnv();                  // kernels are async: wait before stopping the clock
        double dt = nowSec() - t0;

        total += dt;
        if (dt < best) best = dt;
        printf("  rep %d: %8.3f ms\n", r, dt * 1e3);
    }
    
    // Sanity check: QFT is unitary, so the total probability must stay 1
    printf("  total prob    : %.12f\n", calcTotalProb(q));

    // Rough bandwidth figure: assume every gate reads and writes the whole
    // state vector once (an upper bound - controlled phases touch only part of it)
    double bytesPerGate = 2.0 * (double)(1LL << n) * sizeof(qcomp);
    printf("\n  gates per QFT : %ld\n", gates);
    printf("  best / mean   : %.3f / %.3f ms\n", best * 1e3, total / reps * 1e3);
    printf("  time per gate : %.3f us\n", best / gates * 1e6);

    // clean up
    destroyQureg(q);
    finalizeQuESTEnv();
    return 0;
}
