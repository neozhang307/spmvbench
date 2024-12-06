# DASP
Specific Dense Matrix Multiply-Accumulate Units Accelerated General Sparse Matrix-Vector Multiplication


## Introduction

Sparse matrix-vector multiplication (SpMV) plays a key role in computational science and engineering, graph processing and machine learning applications. In this work, we propose DASP, a new algorithm using specific dense MMA units for accelerating the compute part of general SpMV. We analyze the row-wise distribution of nonzeros and group the rows into three categories containing long, medium and short rows, respectively. We then organize them into small blocks of proper sizes to meet the requirement of MMA computation. For the three categories, DASP offers different strategies to complete SpMV by efficiently utilizing the MMA units.

## Artifact Dependencies & Requirements

### Description of the hardware resources
- GPU: NVIDIA A100 GPU(FP64 Tensor Core, FP16 Tensor Core) and NVIDIA H800 GPU(FP64 Tensor Core). Overall, the GPU being used should be one or two  NVIDIA GPUs with FP64 Tensor Core and FP16 Tensor Core.

- CPU: Intel Xeon Silver 4210 CPU. Please use multicore CPU.

- Disk Space: at least 750GB (to store the experiment input dataset).

### Description of the operating systems
- Any Linux system that can support CUDA v12.0 or above and GCC v12.0 or above.

### Software librarie
- NVIDIA CUDA Toolkit v12.0 or above; GCC v12.0 or above; Python 3.9. 

- Python libraries: Pandas, Matplotlib, Numpy, Scipy.

### Input datase
Our experimental dataset includes all 2893 matrices in the SuiteSparse Matrix Collection which is publicly available(https://sparse.tamu.edu/about).
The SuiteSparse Matrix Collection is so far the best collection that resolves the sparse matrix test problems [Duff et al., Sparse Matrix Test Problems, ACM TOMS, 1989]. SuiteSparse includes matrices from 264 application domains, and has been cited more than 4,300 times. Using matrices from SuiteSparse as input dataset of DASP will make our work more understandable and reproducible.

## Artifact Installation & Deployment Process

### Installation and compilation (3 minutes)
Download our artifacts from the link(https://doi.org/10.5281/zenodo.8084940). Once downloaded locally, use the command to unzip it.

`unzip DASP.zip`

DASP requires both NVCC version and GCC version 12.0 and above. If the default compiler version of the current environment does not meet the requirements, users need to manually change the compiler path in the DASP/Makefile file:

`line2: NVCC = /Your_CUDA_path/bin/nvcc`

`line4: GCC = /Your_GCC_path/bin/gcc`

If you are using a machine other than the A100, change the arch number at line6. After that, in the artifact directory DASP/, compile the project. 

`make`

Then you can get the executable files **spmv_double** and **spmv_half**.

Run the following commands:

`cd test`

`bash run_test.sh`

This test demonstrates that the artifacts are available and verifies the correctness of the results with the matrix 'cop20k_A' as the input data.


### Preparation for Dataset (35 hours or more)
Our experimental dataset includes all 2893 matrices in the SuiteSparse Matrix Collection. These matrices need to be downloaded locally via the script provided in the artifact. This process will take approximately 35 hours (at a download speed of 6MB/s, the total size is estimated to be 750GB) or longer.

If users accept the simplified dataset, we have also prepared a small-scale dataset of 309 matrices obtained by random sampling from the SuiteSparse Matrix Collection. The download of the simplified dataset will take approximately 2.5 hours and this dataset is estimated to be 51GB in total size. Using the simplified dataset requires some simple modifications to scripts:

In matrix_download.py: line8 matrix_list1.csv -> matrix_list2.csv

In run_spmv_all.sh: line4 matrix_list1.csv -> matrix_list2.csv

(Optional) Users also can install and use 'axel' tool to download matrices in parallel. In script/matrix_download.py, use line 25 code and comment out code of line 26.

Dataset download command: 

`python3 matrix_download.py`

All matrices will be stored in the directory MM/. If you changed the dataset path, you need to modify the **MM_path** at line 3 in the script run_spmv_all.sh. 


### Preparation for Comparative SpMV Methods

In this paper, for DASP with FP64 double precision, we compared with the cuSPARSE(v12.0), the CSR5 (https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5) and the TileSpMV (https://github.com/SuperScientificSoftwareLaboratory/TileSpMV). For DASP with FP16 half precision, wen just compared with the cuSPARSE(v12.0). The CSR5 and the TileSpMV code have been included in the current artifact.

Compile the CSR5. You can change the compiler path at line 11 in CSR5 Makefile:

`cd spmv_code/CSR5_cuda/`

`make`

Then, compile the TileSpMV. Note the important information we got from the authors of the TileSpMV: it requires that it must be compiled with CUDA v11.1, otherwise many of the matrices will be miscalculated leading to questionable performance records. If the user does not have CUDA v11.1, it is OK to compile directly with the current CUDA version, but the final comparison with DASP will be invalid. You can change the compiler path at line 11 in TileSpMV Makefile. Compile command:

`cd ../TileSpMV/src/`

`make`


### Deployment (20 hours)
After the dataset and the other methods are ready, in the directory script/, run this command:

`bash run.sh`

This script includes tests of double-precision and half-precision SpMV of the target dataset matrices, and will generate a result file and some performance plots similar to those in the paper at the end of the tests. 

The file **result.txt** in DASP/ is the performance analysis results. 

The **a100_f64_scatter.pdf** and the **a100_f16_scatter.pd**f correspond to Figure10 and Figure9(a) in the paper, respectively.  

The **a100_f64_bar.pd**f and the **a100_f16_bar.pdf** correspond to Figure11 in the paper.

The **preprocessing_time.pdf** corresponds to Figure 13 of the paper.

# spmvbench
