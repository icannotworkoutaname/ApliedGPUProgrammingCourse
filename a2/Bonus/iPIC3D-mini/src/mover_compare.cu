/** Standalone mover correctness check: CPU mover_PC() vs GPU mover_PC_gpu().
 *
 * Builds the same initial state as the full simulation (GEM input file),
 * deep-copies the six particle arrays per species, runs both movers on the
 * identical input and compares ALL elements of x/y/z/u/v/w (no sampling):
 *   PASS  iff  |gpu - cpu| <= ATOL + RTOL * |cpu|   for every element.
 * Prints per-species, per-array max abs error and max rel error, plus mover
 * wall/kernel times. Exit code 0 = PASS, 1 = FAIL or runtime error.
 *
 * Usage: bin/mover_compare.out inputfiles/GEM_2D.inp [atol rtol]
 */

#include "Alloc.h"
#include "PrecisionTypes.h"
#include "Parameters.h"
#include "Grid.h"
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"
#include "EMfield.h"
#include "EMfield_aux.h"
#include "Particles.h"
#include "IC.h"
#include "RW_IO.h"
#include "Timing.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {

const char* arr_names[6] = {"x", "y", "z", "u", "v", "w"};

FPpart* arr_ref(struct particles* p, int k) {
    switch (k) {
        case 0: return p->x;
        case 1: return p->y;
        case 2: return p->z;
        case 3: return p->u;
        case 4: return p->v;
        default: return p->w;
    }
}

} // namespace

int main(int argc, char** argv) {
    // tolerances for float arithmetic (FMA / contraction differences);
    // any real bug (wrong index, missing BC, ...) is orders of magnitude larger
    double atol = 1e-4, rtol = 1e-3;
    if (argc > 2) atol = std::atof(argv[2]);
    if (argc > 3) rtol = std::atof(argv[3]);

    std::cout << "=== mover_PC (CPU) vs mover_PC_gpu (GPU) full comparison ===" << std::endl;
    std::cout << "tolerances: atol=" << atol << " rtol=" << rtol << std::endl;

    parameters param;
    // readInputFile rejects argc>=3 unless it is a restart request, so hand it
    // a reduced argv (input file only); atol/rtol stay in this main's argc/argv
    if (argc < 2) {
        std::cerr << "usage: mover_compare <input.inp> [atol] [rtol]" << std::endl;
        return 2;
    }
    char* io_argv[2] = {argv[0], argv[1]};
    readInputFile(&param, 2, io_argv);

    grid grd;
    setGrid(&param, &grd);

    EMfield field;
    field_allocate(&grd, &field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd, &field_aux);

    interpDensSpecies* ids = new interpDensSpecies[param.ns];
    for (int is = 0; is < param.ns; is++)
        interp_dens_species_allocate(&grd, &ids[is], is);

    particles* part = new particles[param.ns];
    for (int is = 0; is < param.ns; is++)
        particle_allocate(&param, &part[is], is);

    initGEM(&param, &grd, &field, &field_aux, part, ids);

    double t_cpu = 0.0, t_gpu = 0.0;
    bool pass = true;
    double worst_abs = 0.0, worst_rel = 0.0;

    for (int is = 0; is < param.ns; is++) {
        particles* p = &part[is];
        const long nop = p->nop;

        // deep copy of the identical initial state for both movers
        FPpart* init[6];
        FPpart* cpu_out[6];
        for (int k = 0; k < 6; k++) {
            init[k]    = new FPpart[nop];
            cpu_out[k] = new FPpart[nop];
            std::memcpy(init[k], arr_ref(p, k), (size_t)nop * sizeof(FPpart));
        }

        // --- CPU mover on a pristine copy ---
        for (int k = 0; k < 6; k++)
            std::memcpy(arr_ref(p, k), init[k], (size_t)nop * sizeof(FPpart));
        double t0 = cpuSecond();
        if (mover_PC(p, &field, &grd, &param) != 0) {
            std::cerr << "CPU mover failed for species " << is << std::endl;
            return 1;
        }
        t_cpu += cpuSecond() - t0;
        for (int k = 0; k < 6; k++)
            std::memcpy(cpu_out[k], arr_ref(p, k), (size_t)nop * sizeof(FPpart));

        // --- GPU mover on the same pristine copy ---
        for (int k = 0; k < 6; k++)
            std::memcpy(arr_ref(p, k), init[k], (size_t)nop * sizeof(FPpart));
        t0 = cpuSecond();
        if (mover_PC_gpu(p, &field, &grd, &param) != 0) {
            std::cerr << "GPU mover failed for species " << is << std::endl;
            return 1;
        }
        t_gpu += cpuSecond() - t0;

        // --- full (non-sampled) comparison of all nop elements of all 6 arrays ---
        std::cout << "--- species " << is << "  (nop=" << nop << ") ---" << std::endl;
        for (int k = 0; k < 6; k++) {
            double max_abs = 0.0, max_rel = 0.0;
            long worst_i = -1;
            for (long i = 0; i < nop; i++) {
                const double cpu_value = (double)cpu_out[k][i];
                const double gpu_value = (double)arr_ref(p, k)[i];
                if (!std::isfinite(cpu_value) || !std::isfinite(gpu_value)) {
                    pass = false;
                    max_abs = max_rel = INFINITY;
                    worst_abs = worst_rel = INFINITY;
                    worst_i = i;
                    continue;
                }
                double d = std::fabs(gpu_value - cpu_value);
                double r = d / std::max(std::fabs(cpu_value), (double)FLT_MIN);
                if (d > max_abs) { max_abs = d; }
                if (r > max_rel) { max_rel = r; worst_i = i; }
                if (d > atol + rtol * std::fabs((double)cpu_out[k][i])) pass = false;
            }
            if (max_abs > worst_abs) worst_abs = max_abs;
            if (max_rel > worst_rel) worst_rel = max_rel;
            std::cout << "  array " << arr_names[k]
                      << ": max_abs_err=" << max_abs
                      << "  max_rel_err=" << max_rel;
            if (worst_i >= 0)
                std::cout << "  (worst at i=" << worst_i << ")";
            std::cout << std::endl;
        }

        for (int k = 0; k < 6; k++) {
            delete[] init[k];
            delete[] cpu_out[k];
        }
    }

    std::cout << "=== mover comparison summary ===" << std::endl;
    std::cout << "worst max_abs_err = " << worst_abs << std::endl;
    std::cout << "worst max_rel_err = " << worst_rel << std::endl;
    std::cout << "CPU mover wall time (s) = " << t_cpu << std::endl;
    std::cout << "GPU mover wall time (s) = " << t_gpu
              << "  (incl. H2D/D2H transfers + allocation)" << std::endl;
    std::cout << "GPU kernel-only time (s) = " << mover_pc_gpu_kernel_time_sec()
              << "  (CUDA events, all species)" << std::endl;
    std::cout << (pass ? "RESULT: PASS" : "RESULT: FAIL") << std::endl;

    // deallocate
    grid_deallocate(&grd);
    field_deallocate(&grd, &field);
    field_aux_deallocate(&grd, &field_aux);
    for (int is = 0; is < param.ns; is++) {
        interp_dens_species_deallocate(&grd, &ids[is]);
        particle_deallocate(&part[is]);
    }
    delete[] ids;
    delete[] part;

    return pass ? 0 : 1;
}
