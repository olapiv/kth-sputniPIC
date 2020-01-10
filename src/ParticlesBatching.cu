#include "ParticlesBatching.h"
#include <cuda.h>
#include <cuda_runtime.h>

/*

size_t queryFreeMemoryOnGPU(void)
{   
        size_t free_byte ;
        size_t total_byte ;

        cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte );
        if ( cudaSuccess != cuda_status ){
            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
            exit(1);
        }

    return (double)free_byte; //return amount of free memory on GPU in bytes
}


int mover_GPU_batch(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***GPU MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;

    // allocate memory for variables on device

    FPpart *x_dev = NULL, *y_dev = NULL, *z_dev = NULL, *u_dev = NULL, *v_dev = NULL, *w_dev = NULL;
    FPinterp *q_dev = NULL;
    FPfield *XN_flat_dev = NULL, *YN_flat_dev = NULL, *ZN_flat_dev = NULL, *Ex_flat_dev = NULL, *Ey_flat_dev = NULL, *Ez_flat_dev = NULL, *Bxn_flat_dev = NULL, *Byn_flat_dev, *Bzn_flat_dev = NULL;

    size_t free_bytes = 0;
    int batch_number = 0;

    free_bytes = queryFreeMemoryOnGPU();

    size_t necessary_bytes = 0;

    const long int to = split_index + MAX_GPU_PARTICILES - 1 < part->npmax - 1 ? split_index + MAX_GPU_PARTICILES - 1 : part->npmax - 1;
    const int n_particles = to - split_index + 1;
    size_t batch_size = (to - split_index + 1) * sizeof(FPpart);


    cudaMalloc(&x_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(x_dev, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&y_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(y_dev, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&z_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(z_dev, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&u_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(u_dev, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&v_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(v_dev, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&w_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(w_dev, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&q_dev, part->npmax * sizeof(FPinterp));
    cudaMemcpy(q_dev, part->q, part->npmax * sizeof(FPinterp), cudaMemcpyHostToDevice);  

    cudaMalloc(&XN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(XN_flat_dev, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&YN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(YN_flat_dev, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&ZN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(ZN_flat_dev, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    
    cudaMalloc(&Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Ex_flat_dev, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Ey_flat_dev, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Ez_flat_dev, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Bxn_flat_dev, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Byn_flat_dev, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Bzn_flat_dev, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);


    const long int to = split_index + MAX_GPU_PARTICILES - 1 < part->npmax - 1 ? split_index + MAX_GPU_PARTICILES - 1 : part->npmax - 1;

    const int n_particles = to - split_index + 1;
    size_t batch_size = (to - split_index + 1) * sizeof(FPpart);        

    FPpart *d_x, *d_y, *d_z, *d_u, *d_v, *d_w;
    cudaMalloc(&d_x, batch_size);
    cudaMalloc(&d_y, batch_size);
    cudaMalloc(&d_z, batch_size);
    cudaMalloc(&d_u, batch_size);
    cudaMalloc(&d_v, batch_size);
    cudaMalloc(&d_w, batch_size);

    //particles
    cudaMemcpy(d_x, part->x+split_index, batch_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_y, part->y+split_index, batch_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_z, part->z+split_index, batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, part->u+split_index, batch_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_v, part->v+split_index, batch_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_w, part->w+split_index, batch_size, cudaMemcpyHostToDevice); 

    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){

        // Call GPU kernel
        single_particle_kernel<<<(part->npmax + TPB - 1)/TPB, TPB>>>(x_dev, y_dev, z_dev,u_dev, v_dev, w_dev, q_dev, XN_flat_dev, YN_flat_dev, ZN_flat_dev, grd->nxn, grd->nyn, grd->nzn, grd->xStart, grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, grd->Lx, grd->Ly, grd->Lz, grd->invVOL, Ex_flat_dev, Ey_flat_dev, Ez_flat_dev, Bxn_flat_dev, Byn_flat_dev, Bzn_flat_dev, param->PERIODICX, param->PERIODICY, param->PERIODICZ, dt_sub_cycling, dto2, qomdt2, part->NiterMover, part->nop);
        cudaDeviceSynchronize();

    } // end of one particle


    // copy memory back to CPU (only the parts that have been modified inside the kernel)

    cudaMemcpy(part->x, x_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, y_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, z_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, u_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, v_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, w_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
        
    cudaMemcpy(field->Ex_flat, Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ey_flat, Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ez_flat, Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bxn_flat, Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Byn_flat, Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bzn_flat, Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    
    // clean up
   
    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaFree(z_dev);
    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(w_dev);

    cudaFree(XN_flat_dev);
    cudaFree(YN_flat_dev);
    cudaFree(ZN_flat_dev);

    cudaFree(Ex_flat_dev);
    cudaFree(Ey_flat_dev);
    cudaFree(Ez_flat_dev);
    cudaFree(Bxn_flat_dev);
    cudaFree(Byn_flat_dev);
    cudaFree(Bzn_flat_dev);

    return(0);
}


void interpP2G_GPU_batch(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{

    FPpart *x_dev = NULL, *y_dev = NULL, *z_dev = NULL, *u_dev = NULL, *v_dev = NULL, *w_dev = NULL;
    FPinterp * q_dev = NULL, *Jx_flat_dev = NULL, *Jy_flat_dev = NULL, *Jz_flat_dev = NULL, *rhon_flat_dev = NULL, *pxx_flat_dev = NULL, *pxy_flat_dev = NULL, *pxz_flat_dev = NULL, *pyy_flat_dev = NULL, *pyz_flat_dev = NULL, *pzz_flat_dev = NULL;
    FPfield *XN_flat_dev = NULL, *YN_flat_dev = NULL, *ZN_flat_dev = NULL;

    cudaMalloc(&x_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(x_dev, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&y_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(y_dev, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&z_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(z_dev, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&u_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(u_dev, part->u, part->npmax* sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&v_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(v_dev, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&w_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(w_dev, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&q_dev, part->npmax * sizeof(FPinterp));
    cudaMemcpy(q_dev, part->q, part->npmax * sizeof(FPinterp), cudaMemcpyHostToDevice);

    cudaMalloc(&Jx_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(Jx_flat_dev, ids->Jx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Jy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(Jy_flat_dev, ids->Jy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Jz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(Jz_flat_dev, ids->Jz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&rhon_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(rhon_flat_dev, ids->rhon_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pxx_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pxx_flat_dev, ids->pxx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pxy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pxy_flat_dev, ids->pxy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pxz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pxz_flat_dev, ids->pxz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pyy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pyy_flat_dev, ids->pyy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pyz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pyz_flat_dev, ids->pyz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pzz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pzz_flat_dev, ids->pzz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&XN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(XN_flat_dev, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&YN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(YN_flat_dev, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&ZN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(ZN_flat_dev, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    interP2G_kernel<<<(part->npmax + TPB - 1)/TPB, TPB>>>(  x_dev, y_dev, z_dev, u_dev, v_dev, w_dev, q_dev, XN_flat_dev, YN_flat_dev, ZN_flat_dev, grd->nxn, grd->nyn, grd->nzn, grd->xStart, grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, grd->invVOL, Jx_flat_dev, Jy_flat_dev, Jz_flat_dev, rhon_flat_dev, pxx_flat_dev , pxy_flat_dev, pxz_flat_dev, pyy_flat_dev, pyz_flat_dev, pzz_flat_dev, part->nop);

    cudaDeviceSynchronize();

    // copy memory back to CPU (only the parts that have been modified inside the kernel)

    cudaMemcpy(ids->Jx_flat, Jx_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jy_flat, Jy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jz_flat, Jz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->rhon_flat, rhon_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxx_flat, pxx_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxy_flat, pxy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxz_flat, pxz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyy_flat, pyy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyz_flat, pyz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pzz_flat, pzz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    
    // clean up

    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaFree(z_dev);
    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(w_dev);
    cudaFree(q_dev);
    cudaFree(XN_flat_dev);
    cudaFree(YN_flat_dev);
    cudaFree(ZN_flat_dev);
    cudaFree(rhon_flat_dev);
    cudaFree(pxx_flat_dev);
    cudaFree(pxy_flat_dev);
    cudaFree(pxz_flat_dev);
    cudaFree(pyy_flat_dev);
    cudaFree(pyz_flat_dev);
    cudaFree(pzz_flat_dev);

}

*/