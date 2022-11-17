KERNEL void addEnergy(GLOBAL const mixed* RESTRICT pmeEnergyBuffer,
                      GLOBAL mixed* RESTRICT energyBuffer,
                      GLOBAL mixed* RESTRICT energyParamDerivs,
                      GLOBAL const real* RESTRICT sliceLambda,
                      int bufferSize) {
    for (int index = GLOBAL_ID; index < bufferSize; index += GLOBAL_SIZE) {
        mixed energy = 0;
        mixed sliceEnergy[NUM_SLICES];
        for (int slice = 0; slice < NUM_SLICES; slice++) {
            sliceEnergy[slice] = pmeEnergyBuffer[index*NUM_SLICES+slice];
            energy += sliceLambda[slice]*sliceEnergy[slice];
        }
        energyBuffer[index] += energy;
        UPDATE_DERIVATIVE_BUFFER
    }
}