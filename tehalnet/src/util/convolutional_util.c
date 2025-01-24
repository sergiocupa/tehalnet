#include "convolutional_util.h"
#include "im2col.h"
#include "col2im.h"



void stream_to_tensor(TensorFloat* tensor)
{
    int batch = tensor->Sizes[0];  // Tamanho do batch
    int in_h = tensor->Sizes[1];  // Altura da entrada
    int in_w = tensor->Sizes[2];  // Largura da entrada
    int in_ch = tensor->Sizes[3];  // Canais de entrada

    int kernel_size = tensor->KernelSize;  // Tamanho do kernel
    int kernel_dila = tensor->KernelDilation;  // Dilatacao do kernel
    int padding     = tensor->Padding;     // Padding aplicado
    int stride      = tensor->Stride;      // Stride utilizado
    int out_h       = (in_h - 1) * stride + kernel_size - 2 * padding;
    int out_w       = (in_w - 1) * stride + kernel_size - 2 * padding;

    int group_ch    = in_ch / 1;           // Grupos de canais, pode ser ajustado se houver `groups`
    int buffer_size = out_h * out_w * kernel_size * kernel_size * group_ch;

    // Ponteiros para os dados de entrada e o buffer stream
    float* data_ptr   = tensor->Data;       // Ponteiro para a matriz original de dados 4D
    float* stream_ptr = tensor->Stream->Data; // Ponteiro para o buffer stream onde estão os dados após im2col + gemm

    // Loop sobre o batch
    for (int i = 0; i < batch; ++i) {
        // Posição no batch atual
        float* input_batch = data_ptr + (i * in_h * in_w * in_ch); // Ponteiro para a matriz 4D de dados
        float* stream_batch = stream_ptr + (i * buffer_size);       // Ponteiro para o buffer stream

        // Aplica col2im para reconstruir o tensor 4D no formato NHWC (batch, altura, largura, canais)
        col2im_cpu_ext_uni(stream_batch, in_ch, in_h, in_w, kernel_size, padding, stride, kernel_dila, input_batch);
    }
}


void tensor_to_stream(TensorFloat* tensor)
{
    int batch   = tensor->Sizes[0];
    int in_h    = tensor->Sizes[1];
    int in_w    = tensor->Sizes[2];
    int filters = tensor->Sizes[3];

    int group_ch = filters / tensor->Groups;
    int k = tensor->KernelSize;
    int p = tensor->Padding;
    int s = tensor->Stride;
    int d = tensor->KernelDilation;

    im2col_cpu_ext(tensor->Data, group_ch, in_h, in_w, k, k, p, p, s, s, d, d, tensor->Stream->Data);
}

