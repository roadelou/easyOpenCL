#include <stdio.h>
#include <stdlib.h>
// #include <iostream> // for standard I/O
// #include <fstream>
// #include <time.h>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 110
#endif

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <string.h>
// #include <math.h>

// Defines len of an easyCL struct, can be re-defined at compile time.
#ifndef EASYCL_REF_LEN
#define EASYCL_REF_LEN 24
#endif




// ##################
// Easy cl struct for code reusability

// typedef enables us to call the struct directly without using the struct keyword.
typedef struct easyCL {
  cl_context context;
  cl_program program;
  cl_kernel kernel;
  cl_command_queue queue;
  size_t len; // = EASYCL_REF_LEN;
  cl_mem buffers[EASYCL_REF_LEN];             // Maximum amount of buffers that can be passed to the opencl code is 24, arbitrary value.
  size_t lenBuffers[EASYCL_REF_LEN];          // The underlying size of each GPU buffer.
  char active[EASYCL_REF_LEN];                // Used to know which events to wait for when starting the kernel in order to avoid undefined behaviors.
  cl_event bufferWriteEvents[EASYCL_REF_LEN]; // Have to be waited for before kernel execution.
  cl_event kernelEvent;                       // Has to be waited for before reading output buffers.
  int kernelEventSet;                         // A value indicating whether the kernel can be read, set to true once the asynchronous run function has retured.
  int error;
} easyCL;





// #################
// Function prototypes

// Note : static functions are kind of private, i.e. not accessible from outside the library.
static const char *getErrorString(cl_int error);
static void checkError(int status, const char *msg);
static void printCLBuildErrors(cl_program program, cl_device_id device);
// static const size_t seekFileSize(const char *name);
// static int readFile(const char *name, const size_t fileSize, char *fileContent);
easyCL compile(const char *fileName);
easyCL setBuffer(easyCL ecl, void *cpuBuffer, size_t lenBuffer, size_t argIndex, int mode);
easyCL readBuffer(easyCL ecl, void *cpuBuffer, size_t argIndex);
easyCL run(easyCL ecl, size_t threadsCount, size_t threadsClusterSize);
int printInfo(easyCL ecl);
int checkCL(easyCL ecl);


// void CPUAdd(size_t len, float *A, float *B, float *C, float *weigths, float *D);
// void random_vector(size_t len, float *A);
//
//
//
//
// // ###################
// // Main function, just put here to test the library. I will have to remove it later on.
//
// int main(int argc, char const *argv[]) {
//
//   clock_t start, end;
//   long int diff;
//
//   start = clock();
//
//   printf("%s\n", "Compiling source code");
//   easyCL exampleCL = compile("vector_add.cl");
//
//   printInfo(exampleCL);
//
//   const size_t N = 50000;
//
//   float input_a[N];
//   float input_b[N];
//   float output[N];
//   float output2[N];
//   float ref_output[N];
//   float ref_output2[N];
//
//   float weights[2] = {1.0,3.0};
//
//   random_vector(N,input_a);
//   random_vector(N,input_b);
//
//   printf("%s\n", "Setting buffers");
//   // printf("%s\n", "Setting first input buffer");
//   exampleCL = setBuffer(exampleCL, (void *)input_a, N*sizeof(float), 0, CL_MEM_READ_ONLY);
//   // printf("%s\n", "Setting second input buffer");
//   exampleCL = setBuffer(exampleCL, (void *)input_b, N*sizeof(float), 1, CL_MEM_READ_ONLY);
//   // printf("%s\n", "Setting output buffer");
//   exampleCL = setBuffer(exampleCL, NULL, N*sizeof(float), 2, CL_MEM_WRITE_ONLY);
//   // printf("%s\n", "Setting weigths buffer");
//   exampleCL = setBuffer(exampleCL, weights, 2*sizeof(float), 3, CL_MEM_READ_ONLY);
//   // printf("%s\n", "Setting second output buffer");
//   exampleCL = setBuffer(exampleCL, NULL, N*sizeof(float), 4, CL_MEM_WRITE_ONLY);
//
//   // checkCL(exampleCL);
//
//   printInfo(exampleCL);
//
//
//   printf("%s\n", "Running kernel");
//   exampleCL = run(exampleCL,N,1);
//
//   printInfo(exampleCL);
//
//   printf("%s\n", "Reading result from kernel");
//   exampleCL = readBuffer(exampleCL, (void *)output, 2);
//   exampleCL = readBuffer(exampleCL, (void *)output2, 4);
//
//   printInfo(exampleCL);
//
//   end = clock();
//   diff = end - start;
//   printf("GPU time : %ld\n", diff);
//
//   start = clock();
//   CPUAdd(N, input_a, input_b, ref_output, weights, ref_output2);
//   end = clock();
//   diff = end - start;
//   printf("CPU time : %ld\n", diff);
//
//   for (size_t i = 0; i < N; i++) {
//     if (output[i]!=ref_output[i]) {
//       printf("Error on index %d for value : %f (ref), %f (GPU), %f (A) et %f (B)\n", i, ref_output[i], output[i], input_a[i], input_b[i]);
//     }
//     // printf("%f - %f - %f - %f\n", ref_output[i], output[i], input_a[i], input_b[i]);
//     if (output2[i]!=ref_output2[i]) {
//       printf("Error on index %d for value : %f (ref2), %f (GPU2), %f (A) et %f (B)\n", i, ref_output2[i], output2[i], input_a[i], input_b[i]);
//     }
//     // printf("%f - %f - %f - %f\n", ref_output2[i], output2[i], input_a[i], input_b[i]);
//
//   }
//   return 0;
// }
//
//
// void random_vector(size_t len, float *A) {
//   for (size_t i = 0; i < len; i++) {
//     A[i] = (float)rand()/(float)(RAND_MAX);
//   }
// }
//
//
// void CPUAdd(size_t len, float *A, float *B, float *C, float *weigths, float *D) {
//   for (size_t i = 0; i < len; i++) {
//     C[i] = weigths[0] * A[i] + weigths[1] * B[i];
//     D[i] = A[i] * B[i];
//   }
// }




// #################################
// Auxiliary functions

// Taken from Sumanta Chaudhuri for his openCL course at Telecom Paris
static const char *getErrorString(cl_int error) {
  switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILE2D";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTI2ES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVAcl_memLID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
  }
}

// Taken from Sumanta Chaudhuri for his openCL course at Telecom Paris
static void checkError(int status, const char *msg) {
  if(status!=CL_SUCCESS) {
    printf("%s: %s\n",msg,getErrorString(status));
  }
}

// Taken from Sumanta Chaudhuri for his openCL course at Telecom Paris
static void printCLBuildErrors(cl_program program,cl_device_id device) {
	printf("%s\n", "Program Build failed");
	size_t length;
	char buffer[2048];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
	printf("--- Build log ---\n%s\n", buffer);
	exit(1);
}

// // Used to create a buffer that will contain the content of the file we are reading from.
// const size_t seekFileSize(const char *name) {
//   size_t size;
//   // unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
//   FILE* fp = fopen(name, "rb");
//   if (!fp) {
//     printf("no such file:%s",name);
//     exit(-1);
//   }
//
//   fseek(fp, 0, SEEK_END);
//   size = ftell(fp);
//   fseek(fp, 0, SEEK_SET);
//
//   fclose(fp);
//
//   return size;
// }

// // Taken from Sumanta Chaudhuri for his openCL course at Telecom Paris
// // fileSize is given by seekFileSize
// int readFile(const char *name, const size_t fileSize, char *fileContent) {
//   FILE* fp = fopen(name, "rb");
//
//   if(!fread(fileContent, fileSize, 1, fp)) {
//     printf("failed to read file\n");
//     return 1;
//   }
//
//   fclose(fp);
//
//   fileContent[fileSize] = '\0'; // trailing null byte
//   return 0;
// }


// ################################
// Main easyOpenCL methods
easyCL setBuffer(easyCL ecl, void *cpuBuffer, size_t lenBuffer, size_t argIndex, int mode) {

  // #########################
  // Creating GPU buffer
  ecl.buffers[argIndex] = clCreateBuffer(ecl.context, mode, lenBuffer, NULL, &ecl.error);
  checkError(ecl.error, "Failed to create GPU buffer");

  ecl.lenBuffers[argIndex] = lenBuffer;

  // ########################
  // Setting value of the buffer if needed
  switch (mode) {
    case CL_MEM_READ_ONLY:  // falling directly to the second case.
    case CL_MEM_READ_WRITE:
    // CL_FALSE here means non blocking call.
      ecl.error = clEnqueueWriteBuffer(ecl.queue, ecl.buffers[argIndex], CL_FALSE,
          0, lenBuffer, cpuBuffer, 0, NULL, &ecl.bufferWriteEvents[argIndex]);
      checkError(ecl.error, "Failed to write buffer from CPU to GPU");
      ecl.active[argIndex] = 1;   // Only buffers that are written to have to be awaited, output buffer events aren't even correctly initialized here.
      break;
  }

  // #####################
  // Setting kernel argument
  ecl.error = clSetKernelArg(ecl.kernel, argIndex, sizeof(cl_mem), &ecl.buffers[argIndex]);
  checkError(ecl.error, "Failed to set kernel argument");

  return ecl;
}


easyCL run(easyCL ecl, size_t threadsCount, size_t threadsClusterSize) {
  // Gathering all active events
  size_t activeEventsCursor = 0;
  cl_event activeEvents[ecl.len];

  for (size_t i = 0; i < ecl.len; i++) {
    if (ecl.active[i] == 1) {
      activeEvents[activeEventsCursor] = ecl.bufferWriteEvents[i];
      activeEventsCursor++;
    }
  }

  // ####################
  // Starting the kernel
  ecl.error = clEnqueueNDRangeKernel(ecl.queue, ecl.kernel, 1, NULL,
      &threadsCount, &threadsClusterSize, activeEventsCursor, activeEvents, &ecl.kernelEvent);
  checkError(ecl.error, "Failed to launch kernel");

  ecl.kernelEventSet = 1;
  return ecl;
}


easyCL readBuffer(easyCL ecl, void *cpuBuffer, size_t argIndex) {
  // ####################
  // Reading the results
  cl_event finishEvent;
  ecl.error = clEnqueueReadBuffer(ecl.queue, ecl.buffers[argIndex], CL_FALSE,
      0, ecl.lenBuffers[argIndex], cpuBuffer, 1, &ecl.kernelEvent, &finishEvent);
  checkError(ecl.error, "Failed to read output from kernel");
  clWaitForEvents(1,&finishEvent);
  // Note that readBuffer is always blocking, unlike setBuffer and run.

  return ecl;
}


int checkCL(easyCL ecl) {
  checkError(ecl.error, "OpenCL raised an error");
  return ecl.error;
}


int printInfo(easyCL ecl) {

  cl_platform_id platform;

  int STRING_BUFFER_LEN = 2048;
  char char_buffer[STRING_BUFFER_LEN];

  clGetPlatformIDs(1, &platform, NULL);

  clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
  printf("[Platform information] %-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
  clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
  printf("[Platform information] %-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
  clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
  printf("[Platform information] %-40s = %s\n", "CL_PLATFORM_VERSION ", char_buffer);
  clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, STRING_BUFFER_LEN, char_buffer, NULL);
  printf("[Platform information] %-40s = %s\n", "CL_PLATFORM_PROFILE", char_buffer);
  clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, STRING_BUFFER_LEN, char_buffer, NULL);
  printf("[Platform information] %-40s = %s\n", "CL_PLATFORM_EXTENSIONS", char_buffer);
  printf("%s\n", "");

  if (checkCL(ecl) == 0) {
    printf("[EasyCL information]   %-40s = %ld\n", "easyCL maximum buffer count", ecl.len);
    char activePrint[3*ecl.len];  // 3 because we assume two digits an a space.
    memset(activePrint, 0, 3*ecl.len);
    size_t activeCursor = 0;
    char smallBuffer[3];    // A small buffer for sprintf, assuming that there aren't more than 99 input buffers.
    for (size_t i = 0; i < ecl.len; i++) {
      if (ecl.active[i]) {
        sprintf(smallBuffer,"%ld",i);
        activePrint[activeCursor] = smallBuffer[0];
        activePrint[activeCursor+1] = smallBuffer[1];
        activePrint[activeCursor+2] = ' ';
        activeCursor += 3;
        // sprintf(activePrint,"%s%ld ",activePrint,i);
      }
      // sprintf(activePrint,"%s%d",activePrint,ecl.active[ecl.len - i - 1]);
    }
    printf("[EasyCL information]   %-40s = %s\n", "easyCL active input buffers", activePrint);
    char kernelStatus[10] = {0};  // Array of fixed length, hence short declaration is allowed
    if (ecl.kernelEventSet) {
      int kernelEventInfo = 0;
      clGetEventInfo(ecl.kernelEvent,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(int),&kernelEventInfo,NULL);
      switch (kernelEventInfo) {
        case CL_QUEUED:
          sprintf(kernelStatus,"%s","Queued");
          break;
        case CL_SUBMITTED:
          sprintf(kernelStatus,"%s","Submitted");
          break;
        case CL_RUNNING:
          sprintf(kernelStatus,"%s","Running");
          break;
        case CL_COMPLETE:
          sprintf(kernelStatus,"%s","Complete");
      }
    }
    else {
      sprintf(kernelStatus,"%s","Not set");
    }
    printf("[EasyCL information]   %-40s = %s\n", "easyCL kernel status", kernelStatus);
  }
  printf("%s\n", "");

  return 0;
}


easyCL compile(const char *fileName) {

  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_context_properties context_properties[] =
  {
       CL_CONTEXT_PLATFORM, 0,
       0    // Trailing 0, cf documentation
  };

  cl_program program;
  cl_command_queue queue;
  cl_kernel kernel;

  easyCL result;

  // int STRING_BUFFER_LEN = 2048;
  // char char_buffer[STRING_BUFFER_LEN];

  clGetPlatformIDs(1, &platform, NULL);

  context_properties[1] = (cl_context_properties)platform;

  result.error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  checkError(result.error, "Failed to get device id");

  context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);

  queue = clCreateCommandQueue(context, device, 0, NULL);

  // printf("%s\n", "Reading source file");
  // size_t sourceSize = seekFileSize(fileName);
  // char *openclProgram[sourceSize];  // Nested array.
  // readFile(fileName, sourceSize, *openclProgram);

  // ###############################
  // Reading source file
  FILE* fp = fopen(fileName, "r");
  if (!fp) {
    printf("No such file : %s",fileName);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size_t sourceSize = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char *openclProgram[sourceSize+1];  // 1 for trailing null byte.
  sourceSize = 0; // cf documentation of getdelim
  ssize_t sourceBytesRead = getdelim((char **) &openclProgram, &sourceSize, '\0', fp);
  fclose(fp);
  if (sourceBytesRead == -1) {
    printf("%s\n", "Failed to read source file");
    result.error = -1;
    return result;
  }

  program = clCreateProgramWithSource(context, 1, (const char **) openclProgram, &sourceSize, &result.error);
  checkError(result.error,"Program creation failed");


  // printf("%s\n", "Compiling program");
  result.error = clBuildProgram(program, 1, (const cl_device_id *) &device, NULL, NULL, NULL);
  checkError(result.error,"Program build failed");
  if(result.error!=CL_SUCCESS) {
    printCLBuildErrors(program,device);
  }

  // printf("%s\n", "Generating kernel");
  kernel = clCreateKernel(program, "k_main", NULL);

  // printf("%s\n", "Initializing result");
  // sending values back to pointers
  result.context = context;
  result.kernel = kernel;
  result.program = program;
  result.queue = queue;
  result.len = EASYCL_REF_LEN;
  // memset(result.buffers,0,result.len);
  memset(result.lenBuffers,0,result.len);
  // memset(result.bufferWriteEvents,0,result.len);
  memset(result.active, 0, result.len);  // Set all events to not defined.
  result.kernelEventSet = 0;
  result.error = 0;

  return result;
}
