#include "Particles.h"
#include "Alloc.h"

#include <iostream>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}

#ifdef __CUDACC__
// ============================================================================
// GPU implementation of mover_PC()
// ----------------------------------------------------------------------------
// Upstream pinned version: KTH-ScaLab/DD2360-HT25 @ 53e7ca59d1904f586b2940d2cfc3f30351873e7b
//
// Design (simple version, as permitted by the assignment):
//   * host wrapper allocates device memory on every call,
//   * copies the six particle arrays and the nine flat field/coordinate arrays
//     H2D, launches one kernel, copies the six particle arrays back D2H, frees.
//   * one thread per particle; each thread runs the particle's subcycles and
//     predictor-corrector iterations sequentially, in the same order as the CPU.
//   * device indexing uses the same flat layout as the host arrays
//     (get_idx(ix,iy,iz,nyn,nzn)); the host pointer chains (***Ex etc.) are
//     never dereferenced on the device.
// ============================================================================

// grid constants passed by value to the kernel (no host struct dereference).
// Types mirror the CPU promotion semantics: grid members xStart/invdx/Lx are
// double, so the index computation and BC arithmetic must be done in double.
struct MoverGPUConst {
    FPpart dt_sub_cycling;
    FPpart dto2;
    FPpart qomdt2;
    double invdx, invdy, invdz;
    FPfield invVOL;
    double xStart, yStart, zStart;
    double Lx, Ly, Lz;
    int n_sub_cycles;
    int NiterMover;
    int nyn, nzn;
    bool PERIODICX, PERIODICY, PERIODICZ;
};

__global__ void mover_pc_kernel(FPpart* x, FPpart* y, FPpart* z,
                                        FPpart* u, FPpart* v, FPpart* w,
                                        const FPfield* __restrict__ Ex_flat,
                                        const FPfield* __restrict__ Ey_flat,
                                        const FPfield* __restrict__ Ez_flat,
                                        const FPfield* __restrict__ Bxn_flat,
                                        const FPfield* __restrict__ Byn_flat,
                                        const FPfield* __restrict__ Bzn_flat,
                                        const FPfield* __restrict__ XN_flat,
                                        const FPfield* __restrict__ YN_flat,
                                        const FPfield* __restrict__ ZN_flat,
                                        MoverGPUConst c, long nop)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nop) return;

    // per-subcycle values; xptilde/yptilde/zptilde are reloaded each subcycle,
    // uptilde/vptilde/wptilde are produced by the predictor-corrector loop
    FPpart xptilde, yptilde, zptilde;
    FPpart uptilde, vptilde, wptilde;

    for (int i_sub = 0; i_sub < c.n_sub_cycles; i_sub++) {
        xptilde = x[i];
        yptilde = y[i];
        zptilde = z[i];
        // predictor-corrector iterations (same order as CPU mover_PC)
        for (int innter = 0; innter < c.NiterMover; innter++) {
            // interpolation G-->P (indices identical to CPU code; CPU computes
            // this in double because the grid members are double)
            int ix = 2 + (int)(((double)x[i] - c.xStart) * c.invdx);
            int iy = 2 + (int)(((double)y[i] - c.yStart) * c.invdy);
            int iz = 2 + (int)(((double)z[i] - c.zStart) * c.invdz);

            // flat device indexing, same layout as host *_flat arrays
            long ixm1 = get_idx(ix - 1, iy,     iz,     c.nyn, c.nzn);
            long iym1 = get_idx(ix,     iy - 1, iz,     c.nyn, c.nzn);
            long izm1 = get_idx(ix,     iy,     iz - 1, c.nyn, c.nzn);
            long i000 = get_idx(ix,     iy,     iz,     c.nyn, c.nzn);

            FPfield xi[2], eta[2], zeta[2];
            xi[0]   = x[i] - XN_flat[ixm1];
            eta[0]  = y[i] - YN_flat[iym1];
            zeta[0] = z[i] - ZN_flat[izm1];
            xi[1]   = XN_flat[i000] - x[i];
            eta[1]  = YN_flat[i000] - y[i];
            zeta[1] = ZN_flat[i000] - z[i];

            FPfield weight[2][2][2];
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * c.invVOL;

            // local electric and magnetic field at the particle
            FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0;
            FPfield Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        long idx = get_idx(ix - ii, iy - jj, iz - kk, c.nyn, c.nzn);
                        Exl += weight[ii][jj][kk] * Ex_flat[idx];
                        Eyl += weight[ii][jj][kk] * Ey_flat[idx];
                        Ezl += weight[ii][jj][kk] * Ez_flat[idx];
                        Bxl += weight[ii][jj][kk] * Bxn_flat[idx];
                        Byl += weight[ii][jj][kk] * Byn_flat[idx];
                        Bzl += weight[ii][jj][kk] * Bzn_flat[idx];
                    }

            FPpart omdtsq = c.qomdt2 * c.qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
            FPpart denom = 1.0 / (1.0 + omdtsq);
            // solve the position equation
            FPpart ut = u[i] + c.qomdt2 * Exl;
            FPpart vt = v[i] + c.qomdt2 * Eyl;
            FPpart wt = w[i] + c.qomdt2 * Ezl;
            FPpart udotb = ut * Bxl + vt * Byl + wt * Bzl;
            // solve the velocity equation
            uptilde = (ut + c.qomdt2 * (vt * Bzl - wt * Byl + c.qomdt2 * udotb * Bxl)) * denom;
            vptilde = (vt + c.qomdt2 * (wt * Bxl - ut * Bzl + c.qomdt2 * udotb * Byl)) * denom;
            wptilde = (wt + c.qomdt2 * (ut * Byl - vt * Bxl + c.qomdt2 * udotb * Bzl)) * denom;
            // update position
            x[i] = xptilde + uptilde * c.dto2;
            y[i] = yptilde + vptilde * c.dto2;
            z[i] = zptilde + wptilde * c.dto2;
        } // end of iteration

        // update the final position and velocity
        u[i] = 2.0 * uptilde - u[i];
        v[i] = 2.0 * vptilde - v[i];
        w[i] = 2.0 * wptilde - w[i];
        x[i] = xptilde + uptilde * c.dt_sub_cycling;
        y[i] = yptilde + vptilde * c.dt_sub_cycling;
        z[i] = zptilde + wptilde * c.dt_sub_cycling;

        ////////// BC (identical logic to CPU mover_PC) //////////
        // X-DIRECTION
        if (x[i] > c.Lx) {
            if (c.PERIODICX) { x[i] = x[i] - c.Lx; }
            else             { u[i] = -u[i]; x[i] = 2 * c.Lx - x[i]; }
        }
        if (x[i] < 0) {
            if (c.PERIODICX) { x[i] = x[i] + c.Lx; }
            else             { u[i] = -u[i]; x[i] = -x[i]; }
        }
        // Y-DIRECTION
        if (y[i] > c.Ly) {
            if (c.PERIODICY) { y[i] = y[i] - c.Ly; }
            else             { v[i] = -v[i]; y[i] = 2 * c.Ly - y[i]; }
        }
        if (y[i] < 0) {
            if (c.PERIODICY) { y[i] = y[i] + c.Ly; }
            else             { v[i] = -v[i]; y[i] = -y[i]; }
        }
        // Z-DIRECTION
        if (z[i] > c.Lz) {
            if (c.PERIODICZ) { z[i] = z[i] - c.Lz; }
            else             { w[i] = -w[i]; z[i] = 2 * c.Lz - z[i]; }
        }
        if (z[i] < 0) {
            if (c.PERIODICZ) { z[i] = z[i] + c.Lz; }
            else             { w[i] = -w[i]; z[i] = -z[i]; }
        }
    } // end of subcycling
}

// ---- CUDA error checking helpers ------------------------------------------
#define CU_CHECK_ERR(call)                                                    \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(_e) << std::endl;       \
            return _e;                                                        \
        }                                                                     \
    } while (0)

// RAII device buffer: frees on every exit path
namespace {
struct DeviceBuffer {
    void* ptr;
    DeviceBuffer() : ptr(nullptr) {}
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    cudaError_t alloc(size_t bytes) { return cudaMalloc(&ptr, bytes); }
};

// RAII CUDA event pair: destroys on every exit path
struct EventGuard {
    cudaEvent_t ev[2];
    int n;
    EventGuard() : n(0) { ev[0] = ev[1] = nullptr; }
    ~EventGuard() { for (int k = 0; k < n; k++) cudaEventDestroy(ev[k]); }
    cudaError_t create() {
        for (int k = 0; k < 2; k++) {
            cudaError_t e = cudaEventCreate(&ev[k]);
            if (e != cudaSuccess) return e;
            n++;
        }
        return cudaSuccess;
    }
};
} // namespace

// accumulated kernel-only time across all calls (seconds), CUDA-event based
static double g_mover_kernel_time_sec = 0.0;

double mover_pc_gpu_kernel_time_sec() { return g_mover_kernel_time_sec; }
void   mover_pc_gpu_reset_kernel_time() { g_mover_kernel_time_sec = 0.0; }

int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    std::cout << "***  GPU MOVER with SUBCYCLYING "<< param->n_sub_cycles
              << " - species " << part->species_ID << " ***" << std::endl;

    const long nop = part->nop;
    if (nop == 0) return 0;  // nothing to move: no transfers, no kernel

    const long nfield = (long)grd->nxn * grd->nyn * grd->nzn;
    const size_t pbytes = (size_t)nop * sizeof(FPpart);
    const size_t fbytes = (size_t)nfield * sizeof(FPfield);

    // one buffer per array so a failure frees everything already allocated
    DeviceBuffer dx, dy, dz, du, dv, dw;
    DeviceBuffer dEx, dEy, dEz, dBx, dBy, dBz, dXN, dYN, dZN;

    CU_CHECK_ERR(dx.alloc(pbytes));
    CU_CHECK_ERR(dy.alloc(pbytes));
    CU_CHECK_ERR(dz.alloc(pbytes));
    CU_CHECK_ERR(du.alloc(pbytes));
    CU_CHECK_ERR(dv.alloc(pbytes));
    CU_CHECK_ERR(dw.alloc(pbytes));
    CU_CHECK_ERR(dEx.alloc(fbytes));
    CU_CHECK_ERR(dEy.alloc(fbytes));
    CU_CHECK_ERR(dEz.alloc(fbytes));
    CU_CHECK_ERR(dBx.alloc(fbytes));
    CU_CHECK_ERR(dBy.alloc(fbytes));
    CU_CHECK_ERR(dBz.alloc(fbytes));
    CU_CHECK_ERR(dXN.alloc(fbytes));
    CU_CHECK_ERR(dYN.alloc(fbytes));
    CU_CHECK_ERR(dZN.alloc(fbytes));

    CU_CHECK_ERR(cudaMemcpy(dx.ptr, part->x, pbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dy.ptr, part->y, pbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dz.ptr, part->z, pbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(du.ptr, part->u, pbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dv.ptr, part->v, pbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dw.ptr, part->w, pbytes, cudaMemcpyHostToDevice));

    // upstream *_flat arrays are already contiguous: copy them directly
    CU_CHECK_ERR(cudaMemcpy(dEx.ptr, field->Ex_flat, fbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dEy.ptr, field->Ey_flat, fbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dEz.ptr, field->Ez_flat, fbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dBx.ptr, field->Bxn_flat, fbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dBy.ptr, field->Byn_flat, fbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dBz.ptr, field->Bzn_flat, fbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dXN.ptr, grd->XN_flat, fbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dYN.ptr, grd->YN_flat, fbytes, cudaMemcpyHostToDevice));
    CU_CHECK_ERR(cudaMemcpy(dZN.ptr, grd->ZN_flat, fbytes, cudaMemcpyHostToDevice));

    MoverGPUConst c;
    c.dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
    c.dto2 = .5 * c.dt_sub_cycling;
    c.qomdt2 = part->qom * c.dto2 / param->c;
    c.invdx = grd->invdx; c.invdy = grd->invdy; c.invdz = grd->invdz;
    c.invVOL = grd->invVOL;
    c.xStart = grd->xStart; c.yStart = grd->yStart; c.zStart = grd->zStart;
    c.Lx = grd->Lx; c.Ly = grd->Ly; c.Lz = grd->Lz;
    c.n_sub_cycles = part->n_sub_cycles;
    c.NiterMover = part->NiterMover;
    c.nyn = grd->nyn; c.nzn = grd->nzn;
    c.PERIODICX = param->PERIODICX;
    c.PERIODICY = param->PERIODICY;
    c.PERIODICZ = param->PERIODICZ;

    const int threads = 256;
    const long blocks = (nop + threads - 1) / threads;

    EventGuard events;
    CU_CHECK_ERR(events.create());
    cudaEvent_t start = events.ev[0], stop = events.ev[1];

    CU_CHECK_ERR(cudaEventRecord(start));
    mover_pc_kernel<<<(unsigned)blocks, threads>>>(
        (FPpart*)dx.ptr, (FPpart*)dy.ptr, (FPpart*)dz.ptr,
        (FPpart*)du.ptr, (FPpart*)dv.ptr, (FPpart*)dw.ptr,
        (const FPfield*)dEx.ptr, (const FPfield*)dEy.ptr, (const FPfield*)dEz.ptr,
        (const FPfield*)dBx.ptr, (const FPfield*)dBy.ptr, (const FPfield*)dBz.ptr,
        (const FPfield*)dXN.ptr, (const FPfield*)dYN.ptr, (const FPfield*)dZN.ptr,
        c, nop);
    CU_CHECK_ERR(cudaGetLastError());
    CU_CHECK_ERR(cudaEventRecord(stop));
    CU_CHECK_ERR(cudaEventSynchronize(stop));

    float ksec = 0.0f;
    CU_CHECK_ERR(cudaEventElapsedTime(&ksec, start, stop));
    g_mover_kernel_time_sec += (double)ksec * 1.e-3;

    CU_CHECK_ERR(cudaMemcpy(part->x, dx.ptr, pbytes, cudaMemcpyDeviceToHost));
    CU_CHECK_ERR(cudaMemcpy(part->y, dy.ptr, pbytes, cudaMemcpyDeviceToHost));
    CU_CHECK_ERR(cudaMemcpy(part->z, dz.ptr, pbytes, cudaMemcpyDeviceToHost));
    CU_CHECK_ERR(cudaMemcpy(part->u, du.ptr, pbytes, cudaMemcpyDeviceToHost));
    CU_CHECK_ERR(cudaMemcpy(part->v, dv.ptr, pbytes, cudaMemcpyDeviceToHost));
    CU_CHECK_ERR(cudaMemcpy(part->w, dw.ptr, pbytes, cudaMemcpyDeviceToHost));

    return 0;  // DeviceBuffer destructors free all device memory here
}
#endif // __CUDACC__
