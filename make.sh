#!/bin/sh

# Useful on my computer but may be futile on another machine.
cp /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 libOpenCL.so
# On fedora the ocl-icd drivers already puts the library libOpenCL.so in /us/lib64/

# # Without GDB
# gcc -I /usr/include -L ./ -lOpenCL -DCL_TARGET_OPENCL_VERSION=110 -o example.elf easyOpenCL.c

# # With GDB
# gcc -g -I /usr/include -L ./ -lOpenCL -DCL_TARGET_OPENCL_VERSION=110 -o example.elf easyOpenCL.c
# # -g means "with debugging symbols"

# Compiling object file
cc -c -Wall -fPIC -I /usr/include -L ./ -lOpenCL -DCL_TARGET_OPENCL_VERSION=110 easyOpenCL.c
cc -shared -Wl,-soname,libEasyOpenCL.so -o libEasyOpenCL.so easyOpenCL.o
# For some reason gcc makes the so executable.
chmod -x libEasyOpenCL.so
cp libEasyOpenCL.so easyOpenCL.h libOpenCL.so ./test/
