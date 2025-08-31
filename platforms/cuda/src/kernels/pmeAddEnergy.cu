KERNEL void addEnergy(GLOBAL mixed* RESTRICT energyBuffer,
#if HAS_DERIVATIVES
                      GLOBAL mixed* RESTRICT energyParamDerivs,
#endif
                      GLOBAL const mixed* RESTRICT pmeEnergyBuffer,
#if USE_LJPME
                      GLOBAL const mixed* RESTRICT ljpmeEnergyBuffer,
#endif
                      GLOBAL const real2* RESTRICT sliceLambdas,
                      int bufferSize) {

    const int index = GLOBAL_ID;
    mixed energy = 0;
    mixed clEnergy[NUM_SLICES];
#if USE_LJPME
    mixed ljEnergy[NUM_SLICES];
#endif
    for (int slice = 0; slice < NUM_SLICES; slice++) {
        clEnergy[slice] = pmeEnergyBuffer[index*NUM_SLICES+slice];
#if USE_LJPME
        ljEnergy[slice] = ljpmeEnergyBuffer[index*NUM_SLICES+slice];
        energy += sliceLambdas[slice].x*clEnergy[slice] + sliceLambdas[slice].y*ljEnergy[slice];
#else
        energy += sliceLambdas[slice].x*clEnergy[slice];
#endif
        }
    energyBuffer[index] += energy;
#if HAS_DERIVATIVES
    ADD_DERIVATIVES
#endif
}