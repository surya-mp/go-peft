#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
dir="$root/backends/cuda"
nvcc -O3 -Xcompiler -fPIC -c "$dir/gopeft_cuda.cu" -o "$dir/gopeft_cuda.o"
ar rcs "$dir/libgopeftcuda.a" "$dir/gopeft_cuda.o"
