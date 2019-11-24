__kernel void k_main(__global const float *input_a,
                      __global const float *input_b,
                      __global float *restrict output) {
  // Getting important values
  size_t Nsquare = get_global_size(0);
  size_t N = convert_int(sqrt((float) Nsquare));
  size_t id = get_global_id(0);
  size_t i = convert_int(id/N);
  size_t j = id%N;

  // Actually computing the matrix multiplication
  // Note that id = i * N + j
  output[id] = 0;
  for (size_t k = 0; k < N; k++) {
    output[id] += input_a[i*N + k] * input_b[k*N + j];
  }
}
