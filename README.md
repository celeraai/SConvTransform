SConvTransform
================================================================================

> SConvTransform is an extension of the MLIR [Transform Dialect][1] that
> introduces new tranform ops to translate Convolutions from the [Linalg
> Dialect][2] into a tiled and packed loop nest that invokes an optimized OpenBLAS
> microkernel.

Follow the instructions for [Installation](#Installation) and [Usage](#Usage).
Please refer to [References](#References) for further references and
instructions for citing this work.

Overview
--------------------------------------------------------------------------------

SConvTransform introduces two new operations to the Transform dialect. The first
one, called `structured.sconv`, is responsible for lowering a
`linalg.conv_2d_nchw_fchw` operation to an optimized loop nest. It works by
first invoking a *Convolution Slicing Analysis* (CSA) pass that takes in the
shape of the convolution as well as information about the target architecture
(i.e. cache size and latency) as well as microkernel size. The analysis produces
as output the number of tiles that fit inside each cache level as well as the
channel grouping factor. These scalars guide the tiling and packing that happens
next. In the first level of tiling the optimized program packs slices of the
input and filter tensors on the cache hierarchy. These tiles are then iterated
over as sub-tiles in the second tiling level and a GEMM operation is left inside
the body of the innermost loop represented as a `linalg.generic` operation.
After running the bufferization pass on this intermediate program, we can invoke
the second operation introduced by SConvTransform, called `lower.to_blas`, that
substitutes the generic microkernel by a call to an outer product GEMM
microkernel from the OpenBLAS library.

Installation
--------------------------------------------------------------------------------

In order to compile SConv from source, you must first have an LLVM build ready
with MLIR enabled. If you don't have one, follow the user-guide [Building LLVM
with CMake][3]. After that, clone the SConvTransform repository and run the
following commands inside the root directory:

```bash
export LLVM_BUILD_DIR="/path/to/llvm-project/build"
make configure && make build
```

After the build has finished, the `sconv-opt` tool will be available at
`build/bin` from the root directory.

Usage
--------------------------------------------------------------------------------

Given a Transform IR file `transform.mlir` and a payload file `payload.mlir`,
SConvTransform can be invoked using the following command:

```bash
build/bin/sconv-opt -transform=transform.mlir payload.mlir > output.mlir
```

### Transform IR

The following Transform IR can be used as a starting point. The first operation
matches all convolutions in the function and return a handle for them. This
handle is passed to the `structured.sconv` transform that returns handles to the
generic microkernels and loops. Notice the microkernel shape passed via the
`mK_info` attribute. Next we bufferize the entire function and apply some
simplification passes. At this point, all of the existing handles are
invalidated because of the bufferization pass. Therefore we match all operations
with the `microkernel` attribute, iterate over all the individual operations,
and lower them to OpenBLAS calls.

```mlir
module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {

    %convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">

    // Apply SConv algorithm
    %ukernels_0, %loops = transform.structured.sconv %convs
      { mK_info = [16, 4] }
      : (!transform.op<"linalg.conv_2d_nchw_fchw">) 
      -> (!transform.op<"linalg.generic">, !transform.any_op)

    %bufferized = transform.bufferization.one_shot_bufferize %arg0
      { bufferize_function_boundaries = true }
      : (!transform.any_op) -> !transform.any_op

    %funcs = transform.structured.match ops{["func.func"]} in %bufferized 
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %funcs {
      transform.apply_patterns.memref.extract_address_computations 
      transform.apply_patterns.memref.expand_strided_metadata 
    } : !transform.any_op

    transform.apply_cse to %bufferized : !transform.any_op
    transform.apply_dce to %bufferized : !transform.any_op

    %ukernels_1 = transform.structured.match
      ops{["linalg.generic"]}
      attributes {microkernel} in %bufferized
      : (!transform.any_op) -> !transform.op<"linalg.generic">
 
    transform.foreach %ukernels_1 : !transform.op<"linalg.generic"> -> !transform.any_op {
    ^bb1(%op: !transform.op<"linalg.generic">):
    // Lower each microkernel individually
      %lowered = transform.lower.to_blas "sgemm_blas_kernel", %op
        : (!transform.op<"linalg.generic">) -> (!transform.any_op)
      transform.yield %lowered : !transform.any_op
    }

    transform.yield
  }
}
```

References
--------------------------------------------------------------------------------

You can learn more about the algorithm and implementation details by reading the
following papers:

- Victor Ferrari, Rafael Sousa, Marcio Pereira, João P. L. De Carvalho, José
  Nelson Amaral, José Moreira, and Guido Araujo. 2023. **Advancing Direct
  Convolution Using Convolution Slicing Optimization and ISA Extensions**. ACM
  Trans. Archit. Code Optim. 20, 4, Article 54 (December 2023), 26 pages.
  https://doi.org/10.1145/3625004

- Victor Ferrari, Lucas Albrenga, Gustavo Leite, Marcio Pereira, and Guido
 Araujo. 2025. **Using MLIR Transform to Design Sliced Convolution Algorithm**.
 Unpublished. [PDF](docs/assets/sconvtransform-2025.pdf).

If you use this work in academic reserach, please cite the following
publication:

```bibtex
@article{10.1145/3625004,
  author = {Ferrari, Victor and Sousa, Rafael and Pereira, Marcio and L. De Carvalho, Jo\~{a}o P. and Amaral, Jos\'{e} Nelson and Moreira, Jos\'{e} and Araujo, Guido},
  title = {Advancing Direct Convolution Using Convolution Slicing Optimization and ISA Extensions},
  year = {2023},
  issue_date = {December 2023},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {20},
  number = {4},
  issn = {1544-3566},
  url = {https://doi.org/10.1145/3625004},
  doi = {10.1145/3625004},
  journal = {ACM Trans. Archit. Code Optim.},
  month = dec,
  articleno = {54},
  numpages = {26},
  keywords = {Convolution, packing, cache blocking, compilers}
}
```

[1]: https://mlir.llvm.org/docs/Dialects/Transform/
[2]: https://mlir.llvm.org/docs/Dialects/Linalg/
[3]: https://llvm.org/docs/CMake.html
