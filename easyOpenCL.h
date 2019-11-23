
// Compiles an openCL program from source and returns the handle used for the execution
easyCL compile(const char *fileName);

// Sets an input or output buffer for the compiled openCL program
easyCL setBuffer(easyCL ecl, void *cpuBuffer, size_t lenBuffer, size_t argIndex, int mode);

// Reads an output buffer from the GPU once the openCL kernel is done running.
easyCL readBuffer(easyCL ecl, void *cpuBuffer, size_t argIndex);

// Runs the current openCL kernel with the supplied arguments.
easyCL run(easyCL ecl, size_t threadsCount, size_t threadsClusterSize);

// Prints various informations about the current openCL platform and the state of the supplied openCL handle
int printInfo(easyCL ecl);

// Check if an error occured in the openCL handle, does nothing if the state is valid.
int checkCL(easyCL ecl);
