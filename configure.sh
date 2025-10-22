#!/usr/bin/env bash

error() {
    echo "Error: $@" >&2 && exit 1
}

if [[ -z "$LLVM_BUILD_DIR" ]]; then
    error "Variable 'LLVM_BUILD_DIR' is not set."
fi

LLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm"
MLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir"

if [[ ! -d "$LLVM_DIR" ]]; then
    error "Invalid LLVM build directory, could not find '$LLVM_DIR'."
fi

if [[ ! -d "$MLIR_DIR" ]]; then
    error "Invalid LLVM build directory, could not find '$MLIR_DIR'. Did you enable MLIR?"
fi

cmake -S . -B build -G Ninja    \
   -DCMAKE_C_COMPILER=clang     \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DCMAKE_BUILD_TYPE=Debug     \
   -DLLVM_ENABLE_LLD=ON         \
   -DLLVM_DIR="$LLVM_DIR"       \
   -DMLIR_DIR="$MLIR_DIR"
