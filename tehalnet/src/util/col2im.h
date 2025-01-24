#ifndef COL2IM_H
#define COL2IM_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "../../include/tehalnet.h"


    void col2im_add_pixel(float* im, int height, int width, int channels, int row, int col, int channel, int pad, float val);
    void col2im_cpu(float* data_col, int channels, int height, int width, int ksize, int stride, int pad, float* data_im);
    void caffe_set(const int N, const float alpha, float* Y);
    void col2im_cpu_ext(const float* data_col, const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w, float* data_im);
    void col2im_cpu_ext_uni(const float* data_col, const int channels, const int height, const int width, const int kernel, const int padding, const int stride, const int dilation, float* data_im);

#ifdef __cplusplus
}
#endif

#endif /* COL2IM */