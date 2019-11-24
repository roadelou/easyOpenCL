#!/bin/sh
echo "---> Recompiling library <---"
chdir ..
./make.sh
chdir test

echo "---> Compiling test <---"
cc testEasyOpenCL.c -I. -L. -lEasyOpenCL -lOpenCL -lm -DCL_TARGET_OPENCL_VERSION=110 -o test.elf

export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
echo "---> Running test <---"
./test.elf
