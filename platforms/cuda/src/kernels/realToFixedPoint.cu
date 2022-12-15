__device__ inline long long realToFixedPoint(real x) {
    return static_cast<long long>(x * 0x100000000);
}