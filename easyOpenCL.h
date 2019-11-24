#ifndef easyOpenCL_Included
#define easyOpenCL_Included


#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 110
#endif

// include obvious dependancy openCL
#include <CL/cl.h>
#include <CL/cl_ext.h>

// Defines len of an easyCL struct, can be re-defined at compile time.
#ifndef EASYCL_REF_LEN
#define EASYCL_REF_LEN 24
#endif


// typedef enables us to call the struct directly without using the struct keyword.
typedef struct easyCL {
  // OpenCL internal stuff...
  cl_context context;
  cl_program program;
  cl_kernel kernel;
  cl_command_queue queue;

  // EasyCL Wrapper stuff
  size_t len; // = EASYCL_REF_LEN;
  cl_mem buffers[EASYCL_REF_LEN];             // Maximum amount of buffers that can be passed to the opencl code is 24, arbitrary value.
  size_t lenBuffers[EASYCL_REF_LEN];          // The underlying size of each GPU buffer.
  char active[EASYCL_REF_LEN];                // Used to know which events to wait for when starting the kernel in order to avoid undefined behaviors.
  cl_event bufferWriteEvents[EASYCL_REF_LEN]; // Have to be waited for before kernel execution.
  cl_event kernelEvent;                       // Has to be waited for before reading output buffers.
  int kernelEventSet;                         // A value indicating whether the kernel can be read, set to true once the asynchronous run function has retured.
  int error;                                  // A status flag used by CheckCL
} easyCL;

// Compiles an openCL program from source and returns the handle used for the execution
extern easyCL compile(const char *fileName);

// Sets an input or output buffer for the compiled openCL program
extern easyCL setBuffer(easyCL ecl, void *cpuBuffer, size_t lenBuffer, size_t argIndex, int mode);

// Reads an output buffer from the GPU once the openCL kernel is done running.
extern easyCL readBuffer(easyCL ecl, void *cpuBuffer, size_t argIndex);

// Sets all buffers back to 0 and prepares the kernel for new execution
extern easyCL resetBuffers(easyCL);

// Runs the current openCL kernel with the supplied arguments.
extern easyCL run(easyCL ecl, size_t threadsCount, size_t threadsClusterSize);

// Prints various informations about the current openCL platform and the state of the supplied openCL handle
extern int printInfo(easyCL ecl);

// Check if an error occured in the openCL handle, does nothing if the state is valid.
extern int checkCL(easyCL ecl);

#endif
