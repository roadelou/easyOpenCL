void multiply(__global const float *input_x, __global const float *input_y, __global float *output);


__kernel void k_main(__global const float *x,
                      __global const float *y,
                      __global float *restrict z,
                      __global float *restrict weights,
                      __global float *restrict w)
{
  size_t current_thread = get_global_id(0);
  z[current_thread] = weights[0]*x[current_thread] + weights[1]*y[current_thread];
  multiply(x,y,w);
}


void multiply(__global const float *input_x, __global const float *input_y, __global float *output) {
  size_t current_thread = get_global_id(0);
  output[current_thread] = input_x[current_thread]*input_y[current_thread];
}
