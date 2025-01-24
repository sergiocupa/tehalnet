#include "GeneralMatrixMultiplier.h"
#include <intrin.h>

#define TILE_M 4 // 4 ops
#define TILE_N 16 // AVX2 = 2 ops * 8 floats
#define TILE_K 16 // loop
#define PUT_IN_REGISTER register

// TO-DO: Implementar uso de GPU
//        Usar por carregamento de biblioteca externa LOADER
//
//void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA,
//    float* A_gpu, int lda,
//    float* B_gpu, int ldb,
//    float BETA,
//    float* C_gpu, int ldc)
//{
//    cublasHandle_t handle = blas_handle();
//    cudaError_t stream_status = (cudaError_t)cublasSetStream(handle, get_cuda_stream());
//    CHECK_CUDA(stream_status);
//
//    cudaError_t status = (cudaError_t)cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),(TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
//
//    CHECK_CUDA(status);
//}
//
//void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
//    float* A, int lda,
//    float* B, int ldb,
//    float BETA,
//    float* C, int ldc)
//{
//    float* A_gpu = cuda_make_array(A, (TA ? lda * K : lda * M));
//    float* B_gpu = cuda_make_array(B, (TB ? ldb * N : ldb * K));
//    float* C_gpu = cuda_make_array(C, ldc * M);
//
//    gemm_ongpu(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);
//
//    cuda_pull_array(C_gpu, C, ldc * M);
//    cuda_free(A_gpu);
//    cuda_free(B_gpu);
//    cuda_free(C_gpu);
//}


void gemm_nn(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) 
    {
        for (k = 0; k < K; ++k) 
        {
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) 
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_nn_fast(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc)
{
    int i, j, k;
    #pragma omp parallel for
    for (i = 0; i < M; ++i) 
    {
        for (k = 0; k < K; ++k)
        {
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) 
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}
void gemm_nt(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            PUT_IN_REGISTER float sum = 0;
            for (k = 0; k < K; ++k) 
            {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) 
    {
        for (k = 0; k < K; ++k)
        {
            PUT_IN_REGISTER float A_PART = ALPHA * A[k * lda + i];
            for (j = 0; j < N; ++j) 
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j) 
        {
            PUT_IN_REGISTER float sum = 0;
            for (k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

void gmm_cpu(bool transposed_a, bool transposed_b, int m, int n, int k, float multiplier_ab, float* a, int displacement_a, float* b, int displacement_b, float multiplier_c, float* c, int displacement_c)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    if (multiplier_c != 1)
    {
        int i, j;
        for (i = 0; i < m; ++i)
        {
            for (j = 0; j < n; ++j)
            {
                c[i * displacement_c + j] *= multiplier_c;
            }
        }
    }

    int t = 0;
    #pragma omp parallel for
    for (t = 0; t < m; ++t) 
    {
        if (!transposed_a && !transposed_b)
        {
            gemm_nn(1, n, k, multiplier_ab, a + t * displacement_a, displacement_a, b, displacement_b, c + t * displacement_c, displacement_c);
        }
        else if (transposed_a && !transposed_b)
        {
            gemm_tn(1, n, k, multiplier_ab, a + t, displacement_a, b, displacement_b, c + t * displacement_c, displacement_c);
        }
        else if (!transposed_a && transposed_b)
        {
            gemm_nt(1, n, k, multiplier_ab, a + t * displacement_a, displacement_a, b, displacement_b, c + t * displacement_c, displacement_c);
        }
        else
        {
            gemm_tt(1, n, k, multiplier_ab, a + t, displacement_a, b, displacement_b, c + t * displacement_c, displacement_c);
        }
    }

    t = 0;
    #pragma omp parallel for
    for (t = 0; t < n; ++t)
    {
        gemm_nn(1, n, k, multiplier_ab, a + t * displacement_a, displacement_a, b, displacement_b, c + t * displacement_c, displacement_c);
    }
}


// M -> Número de linhas da matriz C e da matriz A (se não transposta) ou colunas da matriz A (se transposta).
// N -> Número de colunas da matriz C e da matriz B (se não transposta) ou linhas da matriz B (se transposta).
// K -> Número de colunas da matriz A (se não transposta) ou linhas da matriz A (se transposta) e número de linhas da matriz B (se não transposta) ou colunas da matriz B (se transposta).
// multiplier_ab -> ALPHA
// multiplier_c  => BETA

void general_matrix_multiplier(bool transposed_a, bool transposed_b, int m, int n, int k, float multiplier_ab, float* a, int displacement_a, float* b, int displacement_b, float multiplier_c, float* c, int displacement_c)
{
    gmm_cpu(transposed_a, transposed_b, m, n, k, multiplier_ab, a, displacement_a, b, displacement_b, multiplier_c, c, displacement_c);
}

