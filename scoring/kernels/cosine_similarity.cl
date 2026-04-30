__kernel void cosine_similarity(
    __global const float* query,
    __global const float* docs,
    __global float* out_scores,
    const int dim
) {
    int gid = get_global_id(0);
    float sum = 0.0f;
    int base = gid * dim;
    for (int i = 0; i < dim; i++) {
        sum += query[i] * docs[base + i];
    }
    out_scores[gid] = sum;
}
