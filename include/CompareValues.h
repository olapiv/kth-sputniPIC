#ifndef CompareValues
#define CompareValues

#include <iostream>
#include <math.h>
#include "Alloc.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

void compareValues(struct grid grd, struct interpDensSpecies* idsGPU, struct interpDensSpecies* idsCPU);

#endif
