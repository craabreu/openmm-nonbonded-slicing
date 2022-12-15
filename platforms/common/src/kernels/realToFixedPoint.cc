DEVICE mm_long realToFixedPoint(real x) {
    return static_cast<mm_long>(x * 0x100000000);
}
