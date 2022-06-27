/**
 * This file defines an inline function for type transformation
 */

DEVICE inline mm_long realToFixedPoint(real x) {
    return (mm_long)(x * 0x100000000);
}
