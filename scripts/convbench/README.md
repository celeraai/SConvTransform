## ConvBench

### Overview

**ConvBench** is a comprehensive benchmark designed to evaluate and compare convolution algorithms. It provides a large-scale, realistic dataset and a framework to test optimization passes, particularly, the `SConv` transform.

**Benchmark Dataset**
The benchmark is derived from a comprehensive set of 10,858 convolution instances acquired from 1,280 different deep learning models. For the specific purpose of evaluating `SConv`, grouped convolutions are filtered out, as they use a specialized operation (`linalg.conv_group`) that `SConv` does not target. This filtering results in a dataset of **7,922 convolutions**, broken down as:

  * 6,391 pointwise (1x1 filters)
  * 1,500 regular convolutions
  * 31 non-squared filter convolutions

**Payload Structure**
Each test case is a standalone MLIR payload file. An entry-point `main` function is included in each payload to provide a standardized harness, which:

1.  Creates and initializes the input, weight, and output tensors.
2.  Uses an `scf.for` loop to call the convolution function 30 times, accumulating the total execution time for stable measurement.
3.  Computes the average **GFLOPS/s** performance metric based on the loop's execution time.

### How to Use

The workflow is a two-step process:

1.  **Generate Payloads:**
    Use `gen_payload.sh` to create all the individual MLIR test files from a CSV.

    ```bash
    ./gen_payload.sh <base_payload>.mlir <convset>.csv
    ```

2.  **Run the Benchmark:**
    Use `test_convbench.sh` to execute all generated payloads. It runs each test *with* and *without* your optimization transform to compare the performance.

    ```bash
    ./test_convbench.sh <transform.mlir> <CONVBENCHDIR>
    ```

      * `<transform.mlir>`: Your SConv optimization file.
      * `<CONVBENCHDIR>`: The directory of payloads generated in step 1.
