#include "dasp_f64.h"
#include "GPU_Helpers/warmupHelper.cuh"
#include <cusparse.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>


int verify_new(MAT_VAL_TYPE *cusp_val, MAT_VAL_TYPE *cuda_val, int *new_order, int length)
{
    for (int i = 0; i < length; i ++)
    {
        int cusp_idx = new_order[i];
        if (fabs(cusp_val[cusp_idx] - cuda_val[i]) > 1e-5)
        {
            printf("error in (%d), cusp(%4.2f), cuda(%4.2f),please check your code!\n", i, cusp_val[cusp_idx], cuda_val[i]);
            return -1;
        }
    }
    printf("Y(%d), compute succeed!\n", length);
    return 0;
}


__host__
void cusparse_spmv_all(MAT_VAL_TYPE *cu_ValA, MAT_PTR_TYPE *cu_RowPtrA, int *cu_ColIdxA, 
                       MAT_VAL_TYPE *cu_ValX, MAT_VAL_TYPE *cu_ValY, int rowA, int colA, MAT_PTR_TYPE nnzA,
                       long long int data_origin1, long long int data_origin2, double *cu_time, double *cu_gflops, double *cu_bandwidth1, double *cu_bandwidth2, double *cu_pre)
{
    struct timeval t1, t2;

    MAT_VAL_TYPE *dA_val, *dX, *dY;
    int *dA_cid;
    MAT_PTR_TYPE *dA_rpt;
    MAT_VAL_TYPE alpha = 1.0, beta = 0.0;

    cudaMalloc((void **)&dA_val, sizeof(MAT_VAL_TYPE) * nnzA);
    cudaMalloc((void **)&dA_cid, sizeof(int) * nnzA);
    cudaMalloc((void **)&dA_rpt, sizeof(MAT_PTR_TYPE) * (rowA + 1));
    cudaMalloc((void **)&dX, sizeof(MAT_VAL_TYPE) * colA);
    cudaMalloc((void **)&dY, sizeof(MAT_VAL_TYPE) * rowA);

    cudaMemcpy(dA_val, cu_ValA, sizeof(MAT_VAL_TYPE) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(MAT_PTR_TYPE) * (rowA + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, cu_ValX, sizeof(MAT_VAL_TYPE) * colA, cudaMemcpyHostToDevice);
    cudaMemset(dY, 0.0, sizeof(MAT_VAL_TYPE) * rowA);

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer = NULL;
    size_t               bufferSize = 0;

    gettimeofday(&t1, NULL);
    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, rowA, colA, nnzA, dA_rpt, dA_cid, dA_val,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    

    cusparseCreateDnVec(&vecX, colA, dX, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, rowA, dY, CUDA_R_64F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    // cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double cusparse_pre = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    // printf("cusparse preprocessing time: %8.4lf ms\n", cusparse_pre);
    *cu_pre = cusparse_pre;

    // for (int i = 0; i < 100; ++i)
    // {
    //     cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
    //                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    // }
    WarmupHelper helper;
    helper.warmup(cusparseSpMV, handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    cudaDeviceSynchronize();

    gettimeofday(&t1, NULL);
    for (int i = 0; i < 1000; ++i)
    {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    *cu_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / 1000;
    *cu_gflops = (double)((long)nnzA * 2) / (*cu_time * 1e6);
    *cu_bandwidth1 = (double)data_origin1 / (*cu_time * 1e6); 
    *cu_bandwidth2 = (double)data_origin2 / (*cu_time * 1e6); 
    // printf("cusparse:%8.4lf ms, %8.4lf Gflop/s, %9.4lf GB/s, %9.4lf GB/s\n", *cu_time, *cu_gflops, *cu_bandwidth1, *cu_bandwidth2);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    cudaMemcpy(cu_ValY, dY, sizeof(MAT_VAL_TYPE) * rowA, cudaMemcpyDeviceToHost);

    cudaFree(dA_val);
    cudaFree(dA_cid);
    cudaFree(dA_rpt);
    cudaFree(dX);
    cudaFree(dY);
}

__host__
void cusparse_sorted_spmv_all(MAT_VAL_TYPE *cu_ValA, MAT_PTR_TYPE *cu_RowPtrA, int *cu_ColIdxA, 
                       MAT_VAL_TYPE *cu_ValX, MAT_VAL_TYPE *cu_ValY, int rowA, int colA, MAT_PTR_TYPE nnzA,
                       long long int data_origin1, long long int data_origin2, double *cu_time, double *cu_gflops, double *cu_bandwidth1, double *cu_bandwidth2, double *cu_pre)
{
    struct timeval t1, t2;

    MAT_VAL_TYPE *dA_val, *dX, *dY;
    int *dA_cid;
    MAT_PTR_TYPE *dA_rpt;
    MAT_VAL_TYPE alpha = 1.0, beta = 0.0;

    cudaMalloc((void **)&dA_val, sizeof(MAT_VAL_TYPE) * nnzA);
    cudaMalloc((void **)&dA_cid, sizeof(int) * nnzA);
    cudaMalloc((void **)&dA_rpt, sizeof(MAT_PTR_TYPE) * (rowA + 1));
    cudaMalloc((void **)&dX, sizeof(MAT_VAL_TYPE) * colA);
    cudaMalloc((void **)&dY, sizeof(MAT_VAL_TYPE) * rowA);

    cudaMemcpy(dA_val, cu_ValA, sizeof(MAT_VAL_TYPE) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(MAT_PTR_TYPE) * (rowA + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, cu_ValX, sizeof(MAT_VAL_TYPE) * colA, cudaMemcpyHostToDevice);
    cudaMemset(dY, 0.0, sizeof(MAT_VAL_TYPE) * rowA);

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer = NULL;
    size_t               bufferSize = 0;

    gettimeofday(&t1, NULL);
    cusparseCreate(&handle);
    
    /******************* */
    //Add sortting here
     /******************* */
    // Manual sorting here
    // Count nonzeros per row and create pairs
    std::vector<std::pair<int, int>> row_nnz(rowA);
    for(int i = 0; i < rowA; i++) {
        row_nnz[i] = {i, cu_RowPtrA[i+1] - cu_RowPtrA[i]};
    }
    
    // Sort rows by nonzero count (descending)
    std::sort(row_nnz.begin(), row_nnz.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Create permutation array
    std::vector<int> P(rowA);
    for(int i = 0; i < rowA; i++) {
        P[i] = row_nnz[i].first;
    }
    
    // Create new arrays for sorted matrix
    std::vector<MAT_VAL_TYPE> sorted_vals(nnzA);
    std::vector<int> sorted_cols(nnzA);
    std::vector<MAT_PTR_TYPE> sorted_rowptr(rowA + 1);
    
    // Fill new arrays
    sorted_rowptr[0] = 0;
    MAT_PTR_TYPE current_pos = 0;
    
    for(int i = 0; i < rowA; i++) {
        int old_row = P[i];
        int row_size = cu_RowPtrA[old_row + 1] - cu_RowPtrA[old_row];
        
        // Copy values and column indices
        std::copy(cu_ValA + cu_RowPtrA[old_row], 
                 cu_ValA + cu_RowPtrA[old_row + 1],
                 sorted_vals.data() + current_pos);
        std::copy(cu_ColIdxA + cu_RowPtrA[old_row],
                 cu_ColIdxA + cu_RowPtrA[old_row + 1],
                 sorted_cols.data() + current_pos);
                 
        current_pos += row_size;
        sorted_rowptr[i + 1] = current_pos;
    }
    
    // Copy sorted arrays to device
    cudaMemcpy(dA_val, sorted_vals.data(), sizeof(MAT_VAL_TYPE) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_cid, sorted_cols.data(), sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_rpt, sorted_rowptr.data(), sizeof(MAT_PTR_TYPE) * (rowA + 1), cudaMemcpyHostToDevice);
    /******************* */
    cusparseCreateCsr(&matA, rowA, colA, nnzA, dA_rpt, dA_cid, dA_val,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    /******************* */
    cusparseCreateDnVec(&vecX, colA, dX, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, rowA, dY, CUDA_R_64F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    // cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double cusparse_pre = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    // printf("cusparse preprocessing time: %8.4lf ms\n", cusparse_pre);
    *cu_pre = cusparse_pre;

    WarmupHelper helper;
    helper.warmup(cusparseSpMV, handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    cudaDeviceSynchronize();

    gettimeofday(&t1, NULL);
    for (int i = 0; i < 1000; ++i)
    {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    *cu_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / 1000;
    *cu_gflops = (double)((long)nnzA * 2) / (*cu_time * 1e6);
    *cu_bandwidth1 = (double)data_origin1 / (*cu_time * 1e6); 
    *cu_bandwidth2 = (double)data_origin2 / (*cu_time * 1e6); 
    // printf("cusparse:%8.4lf ms, %8.4lf Gflop/s, %9.4lf GB/s, %9.4lf GB/s\n", *cu_time, *cu_gflops, *cu_bandwidth1, *cu_bandwidth2);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    cudaMemcpy(cu_ValY, dY, sizeof(MAT_VAL_TYPE) * rowA, cudaMemcpyDeviceToHost);
    MAT_VAL_TYPE *temp_ValY = new MAT_VAL_TYPE[rowA];
    for(int i = 0; i < rowA; i++) {
        temp_ValY[P[i]] = cu_ValY[i];
    }
    memcpy(cu_ValY, temp_ValY, sizeof(MAT_VAL_TYPE) * rowA);
    delete[] temp_ValY;
    /******************* */
    //Change cu_ValY to original order
    
    /******************* */
    cudaFree(dA_val);
    cudaFree(dA_cid);
    cudaFree(dA_rpt);
    cudaFree(dX);
    cudaFree(dY);
}
__host__
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Run the code by './spmv_double matrix.mtx $isverify(0 or 1)'. \n");
        return 0;
    }

    // struct timeval t1, t2;
    int rowA, colA;
    MAT_PTR_TYPE nnzA;
    int isSymmetricA;
    MAT_VAL_TYPE *csrValA;
    int *csrColIdxA;
    MAT_PTR_TYPE *csrRowPtrA;

    char *filename;
    filename = argv[1];
    int isverify = atoi(argv[2]);
    int NUM = 4;
    int block_longest = 256;
    double threshold = 0.75;

    printf("\n===%s===\n\n", filename);

    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtrA, &csrColIdxA, &csrValA, filename);
    MAT_VAL_TYPE *X_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * colA);
    initVec(X_val, colA);
    initVec(csrValA, nnzA);

    MAT_VAL_TYPE *dY_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);
    MAT_VAL_TYPE *Y_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);
    int *new_order = (int *)malloc(sizeof(int) * rowA);

    double cu_time = 0, cu_gflops = 0, cu_bandwidth1 = 0, cu_bandwidth2 = 0, cu_pre = 0;
    double cus_time = 0, cus_gflops = 0, cus_bandwidth1 = 0, cus_bandwidth2 = 0, cus_pre = 0;
    long long int data_origin1 = (nnzA + colA + rowA) * sizeof(MAT_VAL_TYPE) + nnzA * sizeof(int) + (rowA + 1) * sizeof(MAT_PTR_TYPE);
    long long int data_origin2 = (nnzA + nnzA + rowA) * sizeof(MAT_VAL_TYPE) + nnzA * sizeof(int) + (rowA + 1) * sizeof(MAT_PTR_TYPE);
    cusparse_spmv_all(csrValA, csrRowPtrA, csrColIdxA, X_val, dY_val, rowA, colA, nnzA, data_origin1, data_origin2, &cu_time, &cu_gflops, &cu_bandwidth1, &cu_bandwidth2, &cu_pre);
    cusparse_sorted_spmv_all(csrValA, csrRowPtrA, csrColIdxA, X_val, dY_val, rowA, colA, nnzA, data_origin1, data_origin2, &cus_time, &cus_gflops, &cus_bandwidth1, &cus_bandwidth2, &cus_pre);
       
    double dasp_pre_time = 0, dasp_spmv_time = 0, dasp_spmv_gflops = 0, dasp_spmv_bandwidth = 0;
    spmv_all(filename, csrValA, csrRowPtrA, csrColIdxA, X_val, Y_val, new_order, rowA, colA, nnzA, NUM, threshold, block_longest, 
             &dasp_pre_time, &dasp_spmv_time, &dasp_spmv_gflops, &dasp_spmv_bandwidth);

    printf("                    pre_time     exe_time       performance\n");
    printf("DASP(Double):    %8.4lf ms  %8.4lf ms  %8.4lf GFlop/s\n", dasp_pre_time, dasp_spmv_time, dasp_spmv_gflops);
    printf("cusparse:        %8.4lf ms  %8.4lf ms  %8.4lf GFlop/s\n", cu_pre, cu_time, cu_gflops);
    printf("cusparse sorted:        %8.4lf ms  %8.4lf ms  %8.4lf GFlop/s\n", cus_pre, cus_time, cus_gflops);

    FILE* fout;
    fout = fopen("data/sorted_f64_record.csv", "a");
    fprintf(fout, "%s,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", filename, rowA, colA, nnzA, dasp_pre_time, dasp_spmv_time, dasp_spmv_gflops, cu_pre, cu_time, cu_gflops, cus_pre, cus_time, cus_gflops);
    fclose(fout);
    
    /* verify the result */
    if (isverify == 1)
    {
        int result = verify_new(dY_val, Y_val, new_order, rowA);
    }

    printf("\n");

    free(X_val);
    free(Y_val);
    free(dY_val);
    free(csrColIdxA);
    free(csrRowPtrA);
    free(csrValA);
    free(new_order);

    return 0;
}