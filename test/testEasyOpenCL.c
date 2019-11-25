#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <easyOpenCL.h>
#include <string.h>
#include <math.h>


void CPUAdd(size_t len, float *A, float *B, float *C, float *weigths, float *D);
void random_vector(size_t len, float *A);
void CPUMultiply(size_t len, float *A, float *B, float *C);


// ###################
// Main function, just put here to test the library. I will have to remove it later on.

int main(int argc, char const *argv[]) {

  clock_t start, end;
  long int diff;

  // ##########################
  // Vector addition
  printf("--- > %s <---\n\n", "Vector Addition");
  printf("%s\n", "Compiling source code");
  easyCL exampleCL = compile("vector_add.cl");

  printInfo(exampleCL);

  const size_t N = 50000;
  const size_t clusterSize = 1;

  float input_a[N];
  float input_b[N];
  float output[N];
  float output2[N];
  float ref_output[N];
  float ref_output2[N];

  float weights[2] = {1.0,3.0};

  for (size_t i = 0; i < 5; i++) {
    printf("Iteration %d\n", i);

    random_vector(N,input_a);
    random_vector(N,input_b);

    start = clock();

    printf("%s\n", "Setting buffers");
    // printf("%s\n", "Setting first input buffer");
    exampleCL = setBuffer(exampleCL, (void *)input_a, N*sizeof(float), 0, CL_MEM_READ_ONLY);
    // printf("%s\n", "Setting second input buffer");
    exampleCL = setBuffer(exampleCL, (void *)input_b, N*sizeof(float), 1, CL_MEM_READ_ONLY);
    // printf("%s\n", "Setting output buffer");
    exampleCL = setBuffer(exampleCL, NULL, N*sizeof(float), 2, CL_MEM_WRITE_ONLY);
    // printf("%s\n", "Setting weigths buffer");
    exampleCL = setBuffer(exampleCL, weights, 2*sizeof(float), 3, CL_MEM_READ_ONLY);
    // printf("%s\n", "Setting second output buffer");
    exampleCL = setBuffer(exampleCL, NULL, N*sizeof(float), 4, CL_MEM_WRITE_ONLY);

    // checkCL(exampleCL);
    // printInfo(exampleCL);


    printf("%s\n", "Running kernel");
    exampleCL = run(exampleCL,&N,&clusterSize);

    // printInfo(exampleCL);

    printf("%s\n", "Reading result from kernel");
    exampleCL = readBuffer(exampleCL, (void *)output, 2);
    exampleCL = readBuffer(exampleCL, (void *)output2, 4);

    printInfo(exampleCL);

    end = clock();
    diff = end - start;
    printf("GPU time : %ld\n", diff);

    printf("%s\n\n", "Resetting buffers");
    exampleCL = resetBuffers(exampleCL);

  }

  start = clock();
  CPUAdd(N, input_a, input_b, ref_output, weights, ref_output2);
  end = clock();
  diff = end - start;
  printf("CPU time : %ld\n", diff);

  for (size_t i = 0; i < N; i++) {
    if (output[i]!=ref_output[i]) {
      printf("Error on index %d for value : %f (ref), %f (GPU), %f (A) et %f (B)\n", i, ref_output[i], output[i], input_a[i], input_b[i]);
    }
    // printf("%f - %f - %f - %f\n", ref_output[i], output[i], input_a[i], input_b[i]);
    if (output2[i]!=ref_output2[i]) {
      printf("Error on index %d for value : %f (ref2), %f (GPU2), %f (A) et %f (B)\n", i, ref_output2[i], output2[i], input_a[i], input_b[i]);
    }
    // printf("%f - %f - %f - %f\n", ref_output2[i], output2[i], input_a[i], input_b[i]);
  }

  printf("%s\n\n\n", "Computation succeed");




  // #################################
  // Matrix multiplication
  printf("--- > %s <---\n\n", "Matrix Multiplication");
  printf("%s\n", "Compiling source code");
  easyCL matrixCL;
  // printInfo(matrixCL);
  matrixCL = compile("matrix_multiply.cl");

  printInfo(matrixCL);

  time_t startT, endT;
  double diffT;

  const size_t n_math = 650;
  const size_t N_mat = n_math*n_math;
  const size_t clusterSize_mat = 1;

  float input_mat_a[N_mat];
  float input_mat_b[N_mat];
  float output_mat[N_mat];
  float ref_output_mat[N_mat];


  for (size_t i = 0; i < 5; i++) {
    printf("Iteration %d\n", i);

    random_vector(N_mat,input_mat_a);
    random_vector(N_mat,input_mat_b);

    start = clock();
    time(&startT);

    printf("%s\n", "Setting buffers");
    // printf("%s\n", "Setting first input buffer");
    matrixCL = setMap(matrixCL, (void *)input_mat_a, N_mat*sizeof(float), 0, CL_MEM_READ_ONLY);
    // printf("%s\n", "Setting second input buffer");
    matrixCL = setMap(matrixCL, (void *)input_mat_b, N_mat*sizeof(float), 1, CL_MEM_READ_ONLY);
    // printf("%s\n", "Setting output buffer");
    matrixCL = setMap(matrixCL, NULL, N_mat*sizeof(float), 2, CL_MEM_WRITE_ONLY);

    printf("%s\n", "Running kernel");
    matrixCL = run(matrixCL,&N_mat,&clusterSize_mat);

    // printInfo(exampleCL);

    printf("%s\n", "Reading result from kernel");
    matrixCL = readMap(matrixCL, (void *)output_mat, 2);

    printInfo(matrixCL);

    end = clock();
    time(&endT);
    diffT = difftime(endT,startT);
    diff = end - start;
    printf("GPU time : %lf - %ld\n", diffT, diff);

    printf("%s\n\n", "Resetting buffers");
    matrixCL = resetBuffers(matrixCL);
  }

  start = clock();
  time(&startT);
  CPUMultiply(N_mat, input_mat_a, input_mat_b, ref_output_mat);
  end = clock();
  time(&endT);
  diffT = difftime(endT,startT);
  diff = end - start;
  printf("CPU time : %lf - %ld\n", diffT, diff);

  int success = 1;
  for (size_t i = 0; i < N_mat; i++) {
    if (output_mat[i]!=ref_output_mat[i]) {
      success = 0;
      printf("Error on index %d for value : %f (ref), %f (GPU), %f (A) et %f (B)\n", i, ref_output_mat[i], output_mat[i], input_mat_a[i], input_mat_b[i]);
    }
  }

  if (success) {
    printf("%s\n", "Computation succeed");
  }
  else {
    printf("%s\n", "Computation failed");
  }


  return 0;
}


void random_vector(size_t len, float *A) {
  for (size_t i = 0; i < len; i++) {
    A[i] = (float)rand()/(float)(RAND_MAX);
  }
}


void CPUAdd(size_t len, float *A, float *B, float *C, float *weigths, float *D) {
  for (size_t i = 0; i < len; i++) {
    C[i] = weigths[0] * A[i] + weigths[1] * B[i];
    D[i] = A[i] * B[i];
  }
}


void CPUMultiply(size_t len, float *A, float *B, float *C) {
  size_t n = (size_t) sqrt(len);
  for (size_t k = 0; k < len; k++) {
    size_t i = (size_t) k/n;
    size_t j = k%n;
    C[k] = 0;
    for (size_t w = 0; w < n; w++) {
      C[k] += A[i*n + w] * B[w*n + j];
    }
  }
}
