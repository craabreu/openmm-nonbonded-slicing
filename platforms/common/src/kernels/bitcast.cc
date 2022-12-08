DEVICE float bitcast_to_float(int in) {
    float out;
    memcpy(&out, &in, sizeof(float));
    return out;
}

DEVICE int bitcast_to_int(float in) {
    int out;
    memcpy(&out, &in, sizeof(int));
    return out;
}