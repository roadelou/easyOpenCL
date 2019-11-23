#!/bin/sh
echo "---> Compiling test <---"
cc testEasyOpenCL.c -v -I. -L. -lEasyOpenCL -lOpenCL -DCL_TARGET_OPENCL_VERSION=110 -o test.elf

export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
echo "---> Running test <---"
./test.elf
