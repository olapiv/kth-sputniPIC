/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    // double iMoverGPU, iInterpGPU, eMoverGPU = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield fieldCPU;
    EMfield fieldGPU;
    field_allocate(&grd,&fieldCPU);
    field_allocate(&grd,&fieldGPU);
    EMfield_aux field_auxCPU;
    EMfield_aux field_auxGPU;
    field_aux_allocate(&grd,&field_auxCPU);
    field_aux_allocate(&grd,&field_auxGPU);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *idsCPU = new interpDensSpecies[param.ns];
    interpDensSpecies *idsGPU = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++) {
        interp_dens_species_allocate(&grd,&idsCPU[is],is);  // Only changes idsCPU
        interp_dens_species_allocate(&grd,&idsGPU[is],is);
    }
    // Net densities
    interpDensNet idnCPU;
    interpDensNet idnGPU;
    interp_dens_net_allocate(&grd,&idnCPU);  // Only changes idnCPU
    interp_dens_net_allocate(&grd,&idnGPU);
    
    // Allocate Particles
    particles *partCPU = new particles[param.ns];
    particles *partGPU = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&partCPU[is],is);
        particle_allocate(&param,&partGPU[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&fieldCPU,&field_auxCPU,partCPU,idsCPU);  // Changes fieldCPU, field_auxCPU, partCPU, idsCPU
    initGEM(&param,&grd,&fieldGPU,&field_auxGPU,partGPU,idsGPU);

    std::cout << " STARTING SIMULATION " << std::endl;

    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idnCPU,idsCPU,&grd,param.ns);  // Only idnCPU & idsCPU is changed
        setZeroDensities(&idnGPU,idsGPU,&grd,param.ns);
        
        
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        for (int is=0; is < param.ns; is++) {
            mover_PC(&partCPU[is],&fieldCPU,&grd,&param);  // Only partCPU is changed
            mover_PC_GPU_basic(&partGPU[is],&fieldGPU,&grd,&param);
        }
        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        
        
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++) {
            interpP2G(&partCPU[is],&idsCPU[is],&grd);  // Only idsCPU is changed
            interpP2G_GPU_basic(&partGPU[is],&idsGPU[is],&grd);
        }
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++) {
            applyBCids(&idsCPU[is],&grd,&param);  // Only idsCPU is changed
            applyBCids(&idsGPU[is],&grd,&param);
        }
        // sum over species
        sumOverSpecies(&idnCPU,idsCPU,&grd,param.ns);  // Only idnCPU is changed
        sumOverSpecies(&idnGPU,idsGPU,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idnCPU.rhon,&grd,&param);  // Only idnCPU is changed
        applyBCscalarDensN(idnGPU.rhon,&grd,&param);
    
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd, &fieldCPU, "cpu");
            VTK_Write_Vectors(cycle, &grd, &fieldGPU, "gpu");

            VTK_Write_Scalars(cycle, &grd,idsCPU,&idnCPU, "cpu");
            VTK_Write_Scalars(cycle, &grd,idsGPU,&idnGPU, "gpu");
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
    
    }  // end of one PIC cycle

    // ------COMPARING RESULTS, OWN CODE--------
    int flat_idx = 0;
    int nxn = grd.nxn;
    int nyn = grd.nyn;
    int nzn = grd.nzn;

    float maxErrorIdsRhon = 0.0f;
    for (int is=0; is < param.ns; is++) {
        for (register int i=0; i < nxn; i++) {
            for (register int j=0; j < nyn; j++) {
                for (register int k=0; k < nzn; k++){        
                    flat_idx = get_idx(i, j, k, nyn, nzn);
                    maxErrorIdsRhon = fmax(maxErrorIdsRhon, fabs(
                        idsCPU[is].rhon[i][j][k] - 
                        idsGPU[is].rhon_flat[flat_idx]
                    ));

                    // TODO: Implement more constants here
                }
            }
        }
    }
    std::cout << "Max error idsrhon: " << maxErrorIdsRhon << std::endl;
    std::cout << "------------" << std::endl;

    float valueFlat;
    float sumIdnRhon = 0.0f;
    float sumIdnRhonFlat = 0.0f;
    float avgIdnRhon = 0.0f;
    float avgIdnRhonFlat = 0.0f;

    float errorIdnRhon = 0.0f;
    float maxErrorIdnRhon = 0.0f;
    float sumErrorIdnRhon = 0.0f;
    for (register int i=0; i < nxn; i++){
        for (register int j=0; j < nyn; j++){
            for (register int k=0; k < nzn; k++){
                flat_idx = get_idx(i, j, k, nyn, nzn);
                valueFlat = idnCPU.rhon_flat[flat_idx];

                sumIdnRhon =+ idnCPU.rhon[i][j][k];
                sumIdnRhonFlat =+ valueFlat;

                errorIdnRhon = fabs(idnCPU.rhon[i][j][k] - valueFlat);
                maxErrorIdnRhon = fmax(maxErrorIdnRhon, errorIdnRhon);
                sumErrorIdnRhon =+ errorIdnRhon;

                // TODO: Implement more constants here
            }
        }
    }
    avgIdnRhon = sumIdnRhon / (nxn * nyn * nzn);
    avgIdnRhonFlat = sumIdnRhonFlat / (nxn * nyn * nzn);
    std::cout << "Avg idnrhon: " << avgIdnRhon << std::endl;
    std::cout << "Avg idnrhonFlat: " << avgIdnRhonFlat << std::endl;

    std::cout << "Max error idnrhon: " << maxErrorIdnRhon << std::endl;
    std::cout << "Avg error idnrhon: " << sumErrorIdnRhon / (nxn * nyn * nzn) << std::endl;

    // ---------------- 

    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&fieldCPU);
    field_deallocate(&grd,&fieldGPU);
    
    interp_dens_net_deallocate(&grd,&idnGPU);
    interp_dens_net_deallocate(&grd,&idnCPU);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&idsCPU[is]);
        particle_deallocate(&partCPU[is]);

        interp_dens_species_deallocate(&grd,&idsGPU[is]);
        particle_deallocate(&partGPU[is]);
    }
    
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


