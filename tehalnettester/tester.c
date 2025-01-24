//#include "../tehalnet/include/tehalnet.h"
#include <stdio.h>
#include <stdlib.h>

#define IS_A_GE_ZERO_AND_A_LT_B(a,b) (unsigned)(a) < (unsigned)(b)

int im2col(const float* data_im, const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w)// , float* data_col)
{
    int count = 0;

    const int output_h     = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w     = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;

    int channel, kernel_row, kernel_col, output_rows, output_col;

    for (channel = channels; channel--; data_im += channel_size)
    {
        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++)
        {
            for (kernel_col = 0; kernel_col < kernel_w; kernel_col++)
            {
                int input_row = -pad_h + kernel_row * dilation_h;

                for (output_rows = output_h; output_rows; output_rows--)
                {
                    if (!IS_A_GE_ZERO_AND_A_LT_B(input_row, height))
                    {
                        for (output_col = output_w; output_col; output_col--)
                        {
                            //*(data_col++) = 0;
                            count++;
                        }
                    }
                    else 
                    {
                        int input_col = -pad_w + kernel_col * dilation_w;

                        for (output_col = output_w; output_col; output_col--)
                        {
                            if (IS_A_GE_ZERO_AND_A_LT_B(input_col, width))
                            {
                                //*(data_col++) = data_im[input_row * width + input_col];
                            }
                            else
                            {
                                //*(data_col++) = 0;
                            }
                            count++;

                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }

    return count;
}



int main()
{
    int width      = 600;
    int height     = 400;
    int channels   = 3;
    int kernel     = 3;
    int stride     = 1;
    int padding    = 1;
    int dilatation = 1;
    int groups     = 1;

    int size     = width * height * channels;
    int out_size = height * width * kernel * kernel * (channels / groups);// *sizeof(float);

    float* data = malloc(size * sizeof(float));

    int cnt1 = im2col(data, channels, height, width, kernel, kernel, 1, 1, 1,1, 1, 1);

    int cnt2 = im2col(data, channels, height, width, kernel, kernel, padding, padding, stride, stride, dilatation, dilatation);

    //teh_init(1, 0, "tehalcuda");

    if (cnt1 > cnt2)
    {
        int abc = 0;
    }


    //byte* aa = Lib.allocate(true, 2222);


    return 0;
}
