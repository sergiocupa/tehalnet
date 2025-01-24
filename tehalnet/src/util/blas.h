#ifndef BLAS_H
#define BLAS_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "../../include/tehalnet.h"


    void axpy_cpu(int N, float ALPHA, float* X, int INCX, float* Y, int INCY);
    void scal_cpu(int N, float ALPHA, float* X, int INCX);

    PLATFORM_API void normalize_cpu(float* input, float* mean, float* variance, int batch, int filters, int spatial, float* output);
    PLATFORM_API void add_bias(float* output, float* biases, int batch, int n, int size);
    PLATFORM_API void scale_bias(float* output, float* scales, int batch, int n, int size);


#ifdef __cplusplus
}
#endif

#endif /* BLAS */