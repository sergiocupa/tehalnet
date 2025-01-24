#include "blas.h"



void axpy_cpu(int N, float ALPHA, float* X, int INCX, float* Y, int INCY)
{
    int i;
    for (i = 0; i < N; ++i) Y[i * INCY] += ALPHA * X[i * INCX];
}

void scal_cpu(int N, float ALPHA, float* X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i) X[i * INCX] *= ALPHA;
}

void normalize_cpu(float* input, float* mean, float* variance, int batch, int filters, int spatial, float* output)
{
    int b, f, i;
    for (b = 0; b < batch; ++b) 
    {
        for (f = 0; f < filters; ++f) 
        {
            for (i = 0; i < spatial; ++i) 
            {
                int index = b * filters * spatial + f * spatial + i;
                output[index] = (input[index] - mean[f]) / (sqrt(variance[f] + .00001f));
            }
        }
    }
}

void add_bias(float* output, float* biases, int batch, int n, int size)
{
    int i, j, b;
    for (b = 0; b < batch; ++b) {
        for (i = 0; i < n; ++i) {
            for (j = 0; j < size; ++j) {
                output[(b * n + i) * size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float* output, float* scales, int batch, int n, int size)
{
    int i, j, b;
    for (b = 0; b < batch; ++b) {
        for (i = 0; i < n; ++i) {
            for (j = 0; j < size; ++j) {
                output[(b * n + i) * size + j] *= scales[i];
            }
        }
    }
}