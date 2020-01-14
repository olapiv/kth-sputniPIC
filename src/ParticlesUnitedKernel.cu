#include "Particles.h"
#include "ParticlesUnitedKernel.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define TPB 64

__global__ void united_kernel(
    // common parameters
    FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w, FPinterp* q, 
    FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat,
    int nxn, int nyn, int nzn,
    double xStart, double yStart, double zStart,
    FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
    int npmax, int NiterMover int stream_offset,

    // single_particle_kernel 
    double Lx, double Ly, double Lz,
    FPfield* Ex_flat, FPfield* Ey_flat, FPfield* Ez_flat,
    FPfield* Bxn_flat, FPfield* Byn_flat, FPfield* Bzn_flat,
    bool PERIODICX, bool PERIODICY, bool PERIODICZ,
    FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2,
    int n_sub_cycles

    // interP2G_kernel 
    FPinterp* Jx_flat, FPinterp* Jy_flat, FPinterp *Jz_flat, FPinterp *rhon_flat,
    FPinterp* pxx_flat, FPinterp* pxy_flat, FPinterp* pxz_flat, FPinterp* pyy_flat, FPinterp* pyz_flat, FPinterp* pzz_flat
){

    //////////////////////////////////
    ///// single_particle_kernel /////
    //////////////////////////////////

    int idx = blockIdx.x * blockDim.x + threadIdx.x + stream_offset; 
    if(idx >= npmax)
    {
        return;
    }
    
    for (int i_sub=0; i_sub <  n_sub_cycles; i_sub++){

         The stream_offset is to account for streaming, by default it is 0 and the result below results in idx = blockIdx.x * blockDim.x + threadIdx.x. 
            When streaming is used, the size of each array (x,y,z etc) is defined in terms of the stream size and hence we need to recalculate the offset to be new_global_offset = stream_offset + blockIdx.x * blockDim.x + threadIdx.x. 

        int flat_idx = 0;

        FPpart omdtsq, denom, ut, vt, wt, udotb;

        // local (to the particle) electric and magnetic field
        FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

        // interpolation densities
        int ix,iy,iz;
        FPfield weight[2][2][2];
        FPfield xi[2], eta[2], zeta[2];

        // intermediate particle position and velocity
        FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

        xptilde = x[idx];
        yptilde = y[idx];
        zptilde = z[idx];

        // calculate the average velocity iteratively
        for(int innter=0; innter < NiterMover; innter++){
            
            // interpolation G-->P
            ix = 2 +  int((x[idx] - xStart)*invdx);
            iy = 2 +  int((y[idx] - yStart)*invdy);
            iz = 2 +  int((z[idx] - zStart)*invdz);

            // calculate weights

            flat_idx = get_idx(ix-1, iy, iz, nyn, nzn);
            xi[0]   = x[idx] - XN_flat[flat_idx];

            flat_idx = get_idx(ix, iy-1, iz, nyn, nzn);
            eta[0]  = y[idx] - YN_flat[flat_idx];

            flat_idx = get_idx(ix, iy, iz-1, nyn, nzn);
            zeta[0] = z[idx] - ZN_flat[flat_idx];

            flat_idx = get_idx(ix, iy, iz, nyn, nzn);
            xi[1]   = XN_flat[flat_idx] - x[idx];
            eta[1]  = YN_flat[flat_idx] - y[idx];
            zeta[1] = ZN_flat[flat_idx] - z[idx];

            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * invVOL;

            // set to zero local electric and magnetic field
            Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

            for (int ii=0; ii < 2; ii++)
                for (int jj=0; jj < 2; jj++)
                    for(int kk=0; kk < 2; kk++){

                        flat_idx = get_idx(ix-ii, iy-jj, iz-kk, nyn, nzn);
                        Exl += weight[ii][jj][kk]*Ex_flat[flat_idx];
                        Eyl += weight[ii][jj][kk]*Ey_flat[flat_idx];
                        Ezl += weight[ii][jj][kk]*Ez_flat[flat_idx];
                        Bxl += weight[ii][jj][kk]*Bxn_flat[flat_idx];
                        Byl += weight[ii][jj][kk]*Byn_flat[flat_idx];
                        Bzl += weight[ii][jj][kk]*Bzn_flat[flat_idx];
                
            } // end interpolation
            
            omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0/(1.0 + omdtsq);

            // solve the position equation
            ut= u[idx] + qomdt2*Exl;
            vt= v[idx] + qomdt2*Eyl;
            wt= w[idx] + qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;

            // solve the velocity equation
            uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
            vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
            wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;

            // update position
            x[idx] = xptilde + uptilde*dto2;
            y[idx] = yptilde + vptilde*dto2;
            z[idx] = zptilde + wptilde*dto2;


        } // end of iteration
        
        // update the final position and velocity
        u[idx]= 2.0*uptilde - u[idx];
        v[idx]= 2.0*vptilde - v[idx];
        w[idx]= 2.0*wptilde - w[idx];
        x[idx] = xptilde + uptilde*dt_sub_cycling;
        y[idx] = yptilde + vptilde*dt_sub_cycling;
        z[idx] = zptilde + wptilde*dt_sub_cycling;


        //////////
        //////////
        ////////// BC

        // X-DIRECTION: BC particles
        if (x[idx] > Lx){
            if (PERIODICX==true){ // PERIODIC
                x[idx] = x[idx] - Lx;
            } else { // REFLECTING BC
                u[idx] = -u[idx];
                x[idx] = 2*Lx - x[idx];
            }
        }

        if (x[idx] < 0){
            if (PERIODICX==true){ // PERIODIC
                x[idx] = x[idx] + Lx;
            } else { // REFLECTING BC
                u[idx] = -u[idx];
                x[idx] = -x[idx];
            }
        }


        // Y-DIRECTION: BC particles
        if (y[idx] > Ly){
            if (PERIODICY==true){ // PERIODIC
                y[idx] = y[idx] - Ly;
            } else { // REFLECTING BC
                v[idx] = -v[idx];
                y[idx] = 2*Ly - y[idx];
            }
        }

        if (y[idx] < 0){
            if (PERIODICY==true){ // PERIODIC
                y[idx] = y[idx] + Ly;
            } else { // REFLECTING BC
                v[idx] = -v[idx];
                y[idx] = -y[idx];
            }
        }

        // Z-DIRECTION: BC particles
        if (z[idx] > Lz){
            if (PERIODICZ==true){ // PERIODIC
                z[idx] = z[idx] - Lz;
            } else { // REFLECTING BC
                w[idx] = -w[idx];
                z[idx] = 2*Lz - z[idx];
            }
        }

        if (z[idx] < 0){
            if (PERIODICZ==true){ // PERIODIC
                z[idx] = z[idx] + Lz;
            } else { // REFLECTING BC
                w[idx] = -w[idx];
                z[idx] = -z[idx];
            }
        }
    }

    ///////////////////////////
    ///// interP2G_kernel /////
    ///////////////////////////

    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];

    // 3-D index of the cell
    int ix, iy, iz, flat_idx;

    ix = 2 + int (floor((x[idx] - xStart) * invdx));
    iy = 2 + int (floor((y[idx] - yStart) * invdy));
    iz = 2 + int (floor((z[idx] - zStart) * invdz));

    // distances from node
    flat_idx = get_idx(ix-1, iy, iz, nyn, nzn);
    xi[0]   = x[idx] - XN_flat[flat_idx];

    flat_idx = get_idx(ix, iy-1, iz, nyn, nzn);
    eta[0]  = y[idx] - YN_flat[flat_idx];

    flat_idx = get_idx(ix, iy, iz-1, nyn, nzn);
    zeta[0] = z[idx] - ZN_flat[flat_idx];

    flat_idx = get_idx(ix, iy, iz, nyn, nzn);
    xi[1]   = XN_flat[flat_idx] - x[idx];
    eta[1]  = YN_flat[flat_idx] - y[idx];
    zeta[1] = ZN_flat[flat_idx] - z[idx];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                // calculate the weights for different nodes
                weight[ii][jj][kk] = q[idx] * xi[ii] * eta[jj] * zeta[kk] * invVOL;


    //////////////////////////
    // add charge density
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&rhon_flat[flat_idx], weight[ii][jj][kk] * invVOL);
            }

    ////////////////////////////
    // add current density - Jx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&Jx_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = v[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&Jy_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = w[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&Jz_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

    ////////////////////////////
    // add pressure pxx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * u[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pxx_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

    ////////////////////////////
    // add pressure pxy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * v[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pxy_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

    /////////////////////////////
    // add pressure pxz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * w[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pxz_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }


    /////////////////////////////
    // add pressure pyy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = v[idx] * v[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pyy_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }


    /////////////////////////////
    // add pressure pyz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) 
                temp[ii][jj][kk] = v[idx] * w[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pyz_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }


    /////////////////////////////
    // add pressure pzz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = w[idx] * w[idx] * weight[ii][jj][kk];

    for (int ii=0; ii < 2; ii++)
        for (int jj=0; jj < 2; jj++)
            for(int kk=0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pzz_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

}
