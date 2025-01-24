#include "convolutional_layer.h"
#include "../util/convolutional_util.h"
#include "activation.h"
#include "../util/box.h"
#include "../util/im2col.h"
#include "../util/col2im.h"
#include "../util/blas.h"
#include "../util/GeneralMatrixMultiplier.h"
#include "../../include/tehalnet.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define _USE_MATH_DEFINES
#include <math.h>



typedef enum ConvolutionalElement
{
    CONVOLUTIONAL_ELEMENT_WEIGTHS    = 0,
    CONVOLUTIONAL_ELEMENT_BIASES     = 1,
    CONVOLUTIONAL_ELEMENT_COL_BUFFER = 2
};



void forward_convolutional_layer(Layer *l, NetworkState state);
void backward_convolutional_layer(Layer l, NetworkState state);
void update_convolutional_layer(Layer l, int batch, float learning_rate_init, float momentum, float decay);
void forward_convolutional_layer_gpu(Layer l, NetworkState state);
void backward_convolutional_layer_gpu(Layer l, NetworkState state);
void update_convolutional_layer_gpu(Layer l, int batch, float learning_rate_init, float momentum, float decay);





Layer* make_convolutional_layer(TensorInfo input, int kernel_size, int kernel_dilation, int output_filters, int padding, int stride, int groups, ActivationOption activation, bool is_train_mode, int steps)
{
    Layer* l = (Layer*)TehLib.allocate(false, sizeof(Layer));
    memset(l, 0, sizeof(Layer));

    int chgr = output_filters % groups;
    if (chgr)
    {
        // Group multiplo da quantidade de filtros
    }

    int batch = input.Sizes[0];
    int in_h  = input.Sizes[1];
    int in_w  = input.Sizes[2];
    int in_ch = input.Sizes[3];

    l->Info.Type                 = LAYER_CONVOLUTIONAL;
    l->Info.Activation           = activation;
    l->Info.Padding              = padding;
    l->Info.Stride               = stride;
    l->Info.Filters              = output_filters;
    l->Info.KernelSize           = kernel_size;
    l->Info.Groups               = groups;
    l->Info.KernelDilation       = kernel_dilation;

    if (l->Info.KernelDilation < 1)  l->Info.KernelDilation = 1;
    if (l->Info.KernelDilation > 6)  l->Info.KernelDilation = 6;

    l->TraiInfo.IsEnabled = is_train_mode;

    int kernel_pow           = l->Info.KernelSize * l->Info.KernelSize;
    int group_input_channels = input.Sizes[3] / groups;
    int group_output_filters = output_filters / groups;
    int output_height        = (input.Sizes[1] + 2 * padding - kernel_size) / stride + 1;
    int output_width         = (input.Sizes[2] + 2 * padding - kernel_size) / stride + 1;
    int steps_cnt            = l->TraiInfo.IsEnabled ? steps : 1;
    
    int out_dim[4] = {input.Sizes[0] * steps_cnt, output_height, output_width, group_output_filters};
    int wei_dim[4] = {output_filters * steps_cnt, kernel_size, kernel_size, group_input_channels};

    l->Input   = TehLib.create_tensor(input.IsDevice, input.Dimensions, input.Sizes);
    l->Output  = TehLib.create_tensor(input.IsDevice, 4, out_dim);

    // Inicializacao do buffer de trabalho
    int si_size = in_h * in_w * kernel_size * kernel_size * in_ch;
    int so_size = in_h * in_w * kernel_size * kernel_size * output_filters;
    TehLib.init_tensor(&l->Input->Stream,  input.IsDevice, 2, (int[2]) { batch, si_size });
    TehLib.init_tensor(&l->Output->Stream, input.IsDevice, 2, (int[2]) { batch, so_size });


    int out_h = (l->Input->Sizes[1] + 2 * padding - kernel_dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (l->Input->Sizes[2] + 2 * padding - kernel_dilation * (kernel_size - 1) - 1) / stride + 1;
    int col_h = group_input_channels * kernel_pow;
    int col_w = out_h * out_w;

    init_tensor_array(&l->Elements);
    tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 4, wei_dim), CONVOLUTIONAL_ELEMENT_WEIGTHS);
    tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, l->Info.Filters), CONVOLUTIONAL_ELEMENT_BIASES);
    tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 2, (int[]) { col_h, col_w }), CONVOLUTIONAL_ELEMENT_COL_BUFFER);

    if (l->TraiInfo.IsEnabled)
    {
        int total_batch = input.Sizes[0] * steps;
        int sizes[4]    = {output_filters * steps_cnt, kernel_size, kernel_size, group_input_channels};

        l->TraiInfo.Delta         = TehLib.create_tensor(input.IsDevice, 1, (int[]) { total_batch * l->Output->TotalSize });
        l->TraiInfo.WeightUpdates = TehLib.create_tensor(input.IsDevice, 4, sizes);
        l->TraiInfo.BiasUpdates   = TehLib.create_tensor(input.IsDevice, 1, l->Info.Filters);
        l->TraiInfo.WeightsEma    = TehLib.create_tensor(input.IsDevice, 4, sizes);
        l->TraiInfo.BiasesEma     = TehLib.create_tensor(input.IsDevice, 1, l->Info.Filters);

        if (l->TraiInfo.Optimizer.Type == OPTIMIZER_ADAM)
        {
            l->TraiInfo.Optimizer.M      = TehLib.create_tensor(input.IsDevice, 4, wei_dim);
            l->TraiInfo.Optimizer.V      = TehLib.create_tensor(input.IsDevice, 4, wei_dim);
            l->TraiInfo.Optimizer.BiasM  = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
            l->TraiInfo.Optimizer.ScaleM = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
            l->TraiInfo.Optimizer.BiasV  = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
            l->TraiInfo.Optimizer.BiasV  = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
        }
    }

    //if (l->BatchNormalize.IsEnabled)// mean, variance, gamma e beta
    //{
    //    if (l->TraiInfo.IsEnabled)
    //    {
    //        l->BatchNormalize.ScalesEma     = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
    //        l->BatchNormalize.ScaleUpdates  = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
    //        l->BatchNormalize.Mean          = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
    //        l->BatchNormalize.Variance      = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
    //        l->BatchNormalize.MeanDelta     = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
    //        l->BatchNormalize.VarianceDelta = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
    //    }

    //    l->BatchNormalize.Scales          = TehLib.create_buffer(input.IsDevice, l->Info.Filters, true, 0);
    //    l->BatchNormalize.RunningMean     = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
    //    l->BatchNormalize.RunningVariance = TehLib.create_buffer(input.IsDevice, l->Info.Filters, false, 0);
    //}

    l->Forward     = forward_convolutional_layer;
    l->Backward    = backward_convolutional_layer;
    l->Update      = update_convolutional_layer;
    l->ForwardGpu  = forward_convolutional_layer_gpu;
    l->BackwardGpu = backward_convolutional_layer_gpu;
    l->UpdateGpu   = update_convolutional_layer_gpu;
}


void forward_convolutional_layer(Layer* layer, TensorFloat* input, TensorFloat* output)
{
    int i, j;

    int batch         = input->Sizes[0];
    int in_h          = input->Sizes[1];
    int in_w          = input->Sizes[2];
    int in_ch         = input->Sizes[3];
    int in_group_ch   = in_ch / layer->Info.Groups;
    int in_batch_size = in_h * in_w * in_ch;// TO-DO: Verificar se, deslocamento para tensor ou stream é o mesmo

    int out_h          = output->Sizes[1];
    int out_w          = output->Sizes[2];
    int out_ch         = output->Sizes[3];
    int out_group_ch   = out_ch / layer->Info.Groups;
    int out_batch_size = out_h * out_w * out_ch;// TO-DO: Verificar se, deslocamento para tensor ou stream é o mesmo

    int k  = layer->Info.KernelSize;
    int kk = k * k;
    int p  = layer->Info.Padding;
    int s  = layer->Info.Stride;
    int d  = layer->Info.KernelDilation;
    int n  = out_h * out_w;
    int m  = layer->Info.Filters / layer->Info.Groups;

    int in_buffer_size  = in_h * in_w * kk * in_group_ch;
    int out_buffer_size = out_h * out_w * kk * out_group_ch;

    for (i = 0; i < batch; ++i)
    {
        float* input_batch  = (input->Data + (i * in_batch_size));
        float* output_batch = (output->Data + (i * out_batch_size));
        float* in_stream    = input->Stream->Data + (i * input->Stream->Sizes[1]);//    TO-DO: input->Stream.Sizes[1] ???
        float* out_stream   = output->Stream->Data + (i * output->Stream->Sizes[1]);//  TO-DO: output->Stream.Sizes[1] ???

        for (j = 0; j < layer->Info.Groups; ++j)
        {
            float* a = layer->Weights->Data + j * layer->Weights->TotalSize / layer->Info.Groups;
            float* b = in_stream + (j * in_buffer_size);
            float* c = out_stream + j * n * m;

            if (!input->IsStreammed)
            {
                float* im = input_batch + j * in_group_ch * in_h * in_w;
                im2col_cpu_ext_uni(im, in_group_ch, in_h, in_w, k, p, s, d, b);
            }

            general_matrix_multiplier(false, false, out_group_ch, n, k, 1, a, k, b, n, 1, c, n);
        }
    }

    if (layer->BuildOutputTensor)// TO-DO: Analizar se pode fazer um metodo que aproveita os loops acima?
    {
        stream_to_tensor(output);
    }

    input->IsStreammed = true;
}



void _forward_convolutional_layer(Layer* l, NetworkState state)
{
    int batchSize = l->Input->Sizes[0];
    int in_h = l->Input->Sizes[1];
    int in_w = l->Input->Sizes[2];
    int inputChannels = l->Input->Sizes[3];

    int ker = l->Info.KernelSize;
    int pad = l->Info.Padding;
    int str = l->Info.Stride;
    int dil = l->Info.KernelDilation;
    int kernel_pow = l->Info.KernelSize * l->Info.KernelSize;

    int out_h = (in_h + 2 * pad - dil * (ker - 1) - 1) / str + 1;
    int out_w = (in_w + 2 * pad - dil * (ker - 1) - 1) / str + 1;

    int group_in_ch = inputChannels / l->Info.Groups;
    int groupOutputFilters = l->Info.Filters / l->Info.Groups;

    // Allocate memory for the im2col matrix
    int col_h = group_in_ch * kernel_pow;
    int col_w = out_h * out_w;

    for (int b = 0; b < batchSize; ++b)
    {
        for (int g = 0; g < l->Info.Groups; ++g)
        {
            const float* inputPtr = l->Input->Data + b * inputChannels * in_h * in_w + g * group_in_ch * in_h * in_w;
            //const float* filterPtr = l->Info.Filters + g * groupOutputFilters * group_in_ch * kernel_pow;

           

            // ???
            //im2col_cpu_ext(inputPtr, group_in_ch, in_h, in_w, ker, ker, pad, pad, str, str, dil, dil, l->ColBuffer->Data);





            //float* outputPtr = l->Output->Data + b * l->Info.Filters * col_w + g * groupOutputFilters * col_w;

            // fff = input->Data + b * inputChannels * inputHeight * inputWidth + g * groupChannels * inputHeight * inputWidth;

            //gemm(0, 0, groupOutputFilters, colWidth, colHeight, 1, filterPtr, colHeight, im2colMatrix, colWidth, 0, outputPtr, colWidth);

            //general_matrix_multiplier(false,false,
        }
    }


    //if (l->BatchNormalize.IsEnabled)
    //{
    //    // TO-DO : implemetar em batchnorm_layer.c
    //    //forward_batchnorm_layer(l, state);
    //}
    //else
    //{
    //    add_bias(l->Output->Data, l->Biases->Data, l->Info.Batch, l->Info.Filters, out_h * out_w);
    //}

    int tb = l->Output->TotalSize * l->Info.Batch;
    int hw = l->Output->Sizes[2] * l->Output->Sizes[1];
    int ch = l->Output->Sizes[0];

    if (l->Info.Activation == ACTIVATION_SWISH)
    {
        //activate_array_swish(l->Output->Data, tb, l->ActivationInput.Data, l->Output->Data);
    }
    else if (l->Info.Activation == ACTIVATION_MISH)
    {
        //activate_array_mish(l->Output->Data, tb, l->ActivationInput.Data, l->Output->Data);
    }
    else if (l->Info.Activation == ACTIVATION_HARD_MISH)
    {
        //activate_array_hard_mish(l->Output->Data, tb, l->ActivationInput.Data, l->Output->Data);
    }
    else if (l->Info.Activation == ACTIVATION_NORM_CHAN)
    {
        activate_array_normalize_channels(l->Output->Data, tb, l->Info.Batch, ch, hw, l->Output->Data);
    }
    else if (l->Info.Activation == ACTIVATION_NORM_CHAN_SOFTMAX)
    {
        activate_array_normalize_channels_softmax(l->Output->Data, tb, l->Info.Batch, ch, hw, l->Output->Data, 0);
    }
    else if (l->Info.Activation == ACTIVATION_NORM_CHAN_SOFTMAX_MAXVAL)
    {
        activate_array_normalize_channels_softmax(l->Output->Data, tb, l->Info.Batch, ch, hw, l->Output->Data, 1);
    }
    else
    {
        activate_array_cpu_custom(l->Output->Data, tb, l->Info.Activation);
    }

    if (l->Info.AssistedExcitation && state.IsTrain)
    {
        assisted_excitation_forward(l, state);
    }

    // TO-DO : implemetar...
   //if (l.antialiasing) {
   //    network_state s = { 0 };
   //    s.train = state.train;
   //    s.workspace = state.workspace;
   //    s.net = state.net;
   //    s.input = l.output;
   //    forward_convolutional_layer(*(l.input_layer), s);
   //    //simple_copy_ongpu(l.outputs*l.batch, l.output, l.input_antialiasing);
   //    memcpy(l.output, l.input_layer->output, l.input_layer->outputs * l.input_layer->batch * sizeof(float));
   //}
}

//
//ConvolutionalLayer* make_convolutional_layer(LayerInfo* param, ConvolutionalLayer* share_layer)
//{
//    int total_batch = param->Batch;// TO-DO: *param->Steps;
//    int i;
//    ConvolutionalLayer* l = (Layer*)TehLib.allocate(false, sizeof(Layer));
//    memset(l, 0, sizeof(ConvolutionalLayer));
//
//    l->Info.Type       = LAYER_CONVOLUTIONAL;
//    l->Info.Activation = param->Activation;
//
//    l->TraiInfo.IsEnabled          = param->IsTrainMode;
//    l->TraiInfo.Batch              = param->Batch;
//    l->TraiInfo.BatchNormalize     = param->BatchNormalize;
//    l->TraiInfo.AssistedExcitation = param->AssistedExcitation;
//    l->TraiInfo.Antialiasing       = param->Antialiasing;
//    l->TraiInfo.Epoch              = 0;
//
//    if (param->Groups < 1) param->Groups = 1;
//
//    const int blur_stride_x = param->StrideX;
//    const int blur_stride_y = param->StrideY;
//
//    int sx = param->StrideX;
//    int sy = param->StrideY;
//    l->Info.Antialiasing    = param->Antialiasing;
//    if (param->Antialiasing)
//    {
//        sx = sy = l->Info.Stride = l->Info.StrideX = l->Info.StrideY = 1; // use stride=1 in host-layer
//    }
//
//    l->Info.WaitStremID        = -1;
//    l->Info.Deform             = param->Deform;
//    l->ShareLayer              = share_layer;
//    l->Info.Index              = param->Index;
//
//
//    init_tensor_info(&param->Input, &l->Info.Input);
//    
//
//
//    l->Info.Groups             = param->Groups;
//    l->Info.Filters            = param->Filters;
//    l->Info.FilterSize         = param->FilterSize;
//    //l->Info.Steps              = param->Steps;
//    l->Info.Stride             = param->Stride;
//    l->Info.StrideX            = sx;
//    l->Info.StrideY            = sy;
//    l->Info.Dilation           = param->Dilation;
//    l->Info.Padding            = param->Padding;
//    l->Info.LearningRateScale  = 1;
//    //l->Info.WeightsPerLayer    = (param->Image.Channels / param->Groups) * param->Filters * param->FilterSize * param->FilterSize;
//
//    if (l->ShareLayer)
//    {
//        if (l->Info.FilterSize != l->ShareLayer->Info.FilterSize ||
//            l->Info.WeightsPerLayer != l->ShareLayer->Info.WeightsPerLayer ||
//            //l->Info.Image.Channels != l->ShareLayer->Info.Image.Channels ||
//            l->Info.Filters != l->ShareLayer->Info.Filters)
//        {
//            error("Layer FilterSize, WeightsPerLayer, Channels or Filters don't match for the ShareLayer");
//        }
//
//        l->Weights       = l->ShareLayer->Weights;
//        l->WeightUpdates = l->ShareLayer->WeightUpdates;
//        l->Biases        = l->ShareLayer->Biases;
//        l->BiasUpdates   = l->ShareLayer->BiasUpdates;
//    }
//    else
//    {
//        l->Weights = TehLib.create_tensor(TehLib.UseDevice, 1, l->Info.WeightsPerLayer);
//        l->Biases  = TehLib.create_tensor(TehLib.UseDevice, 1, l->Info.Filters);
//
//        if (l->TraiInfo.IsEnabled)
//        {
//            l->WeightUpdates = TehLib.create_tensor(TehLib.UseDevice, 1, l->Info.WeightsPerLayer);
//            l->BiasUpdates   = TehLib.create_tensor(TehLib.UseDevice, 1, l->Info.Filters);
//            l->WeightsEma    = TehLib.create_tensor(TehLib.UseDevice, 1, l->Info.WeightsPerLayer);
//            l->BiasesEma     = TehLib.create_tensor(TehLib.UseDevice, 1, l->Info.Filters);
//        }
//    }
//
//    float scale = sqrt(2. / (l->Info.FilterSize * l->Info.FilterSize * l->Info.Image.Channels / l->Info.Groups));
//
//    if (l->Info.Activation == ACTIVATION_NORM_CHAN || l->Info.Activation == ACTIVATION_NORM_CHAN_SOFTMAX || l->Info.Activation == ACTIVATION_NORM_CHAN_SOFTMAX_MAXVAL)
//    {
//        //for (i = 0; i < l.WeightsPerLayer; ++i) l.Weights[i] = 1;   // rand_normal();
//    }
//    else 
//    {
//        //for (i = 0; i < l.WeightsPerLayer; ++i) l.Weights[i] = scale * rand_uniform(-1, 1);   // rand_normal();
//    }
//
//    // Se input for outro tensor?
//    
//    // Info.Input é um template de tensor, para pegar os tamanhos e criar o input
//
//
//    //  
//
//
//    //TehLib.init_tensor_imagem(&l->Output, TehLib.UseDevice, convolutional_out_width(l), convolutional_out_height(l), l->Info.Filters);
//    //TehLib.init_tensor_imagem(&l->Input, TehLib.UseDevice, l->Info.Image.Width, l->Info.Image.Height, l->Info.Image.Channels);
//
//    if (l->TraiInfo.IsEnabled)
//    {
//        TehLib.init_buffer(&l->TraiInfo.Delta, TehLib.UseDevice, total_batch * l->Output.TotalSize);
//    }
//
//    l->Forward     = forward_convolutional_layer;
//    l->Backward    = backward_convolutional_layer;
//    l->Update      = update_convolutional_layer;
//    l->ForwardGpu  = forward_convolutional_layer_gpu;
//    l->BackwardGpu = backward_convolutional_layer_gpu;
//    l->UpdateGpu   = update_convolutional_layer_gpu;
//
//    int buffer_size = 0;// total_batch* l->Output.TotalSize;
//
//    if (l->TraiInfo.BatchNormalize)
//    {
//        if (l->ShareLayer)
//        {
//            l->Scales.Data          = l->ShareLayer->Scales.Data;
//            l->ScaleUpdates.Data    = l->ShareLayer->ScaleUpdates.Data;
//            l->Mean.Data            = l->ShareLayer->Mean.Data;
//            l->Variance.Data        = l->ShareLayer->Variance.Data;
//            l->MeanDelta.Data       = l->ShareLayer->MeanDelta.Data;
//            l->VarianceDelta.Data   = l->ShareLayer->VarianceDelta.Data;
//            l->RollingMean.Data     = l->ShareLayer->RollingMean.Data;
//            l->RollingVariance.Data = l->ShareLayer->RollingVariance.Data;
//        }
//        else
//        {
//            TehLib.init_buffer(&l->Scales, TehLib.UseDevice, l->Info.Filters);
//
//            if (l->TraiInfo.IsEnabled)
//            {
//                TehLib.init_buffer(&l->ScalesEma, TehLib.UseDevice, l->Info.Filters);
//                TehLib.init_buffer(&l->ScaleUpdates, TehLib.UseDevice, l->Info.Filters);
//                TehLib.init_buffer(&l->Mean, TehLib.UseDevice, l->Info.Filters);
//                TehLib.init_buffer(&l->Variance, TehLib.UseDevice, l->Info.Filters);
//                TehLib.init_buffer(&l->VarianceDelta, TehLib.UseDevice, l->Info.Filters);
//            }
//
//            TehLib.init_buffer(&l->RollingMean, TehLib.UseDevice, l->Info.Filters);
//            TehLib.init_buffer(&l->RollingVariance, TehLib.UseDevice, l->Info.Filters);
//        }
//
//        if (l->Info.IsTrainMode)
//        {
//            TehLib.init_buffer(&l->X, TehLib.UseDevice, buffer_size);
//            TehLib.init_buffer(&l->XNorm, TehLib.UseDevice, buffer_size);
//        }
//    }
//
//    if (l->Info.Activation == ACTIVATION_SWISH || l->Info.Activation == ACTIVATION_MISH || l->Info.Activation == ACTIVATION_HARD_MISH)
//    {
//        TehLib.init_buffer(&l->ActivationInput, TehLib.UseDevice, buffer_size);
//    }
//
//    if (l->Info.Optimizer == OPTIMIZER_ADAM)
//    {
//        l->TraiInfo.Optimizer.Type = OPTIMIZER_ADAM;
//
//        TehLib.init_buffer(&l->TraiInfo.Optimizer.M, TehLib.UseDevice, l->Info.WeightsPerLayer);
//        TehLib.init_buffer(&l->TraiInfo.Optimizer.V, TehLib.UseDevice, l->Info.WeightsPerLayer);
//        TehLib.init_buffer(&l->TraiInfo.Optimizer.BiasM, TehLib.UseDevice, l->Info.Filters);
//        TehLib.init_buffer(&l->TraiInfo.Optimizer.BiasV, TehLib.UseDevice, l->Info.Filters);
//        TehLib.init_buffer(&l->TraiInfo.Optimizer.ScaleM, TehLib.UseDevice, l->Info.Filters);
//        TehLib.init_buffer(&l->TraiInfo.Optimizer.ScaleV, TehLib.UseDevice, l->Info.Filters);
//    }
//
//
//    // implementacoes da GPU linhas 675 pra baixo...
//    // Analizar de fazer outro projeto, para somente chamadas para gpu
//
//    if (l->Info.Antialiasing)
//    {
//        l->InputLayer = (Layer*)TehLib.allocate(false, sizeof(Layer));
//
//        int blur_size = 3;
//        int blur_pad  = blur_size / 2;
//        if (l->Info.Antialiasing == 2)
//        {
//            blur_size = 2;
//            blur_pad = 0;
//        }
//
//        LayerInfo inpar;
//        memset(&inpar,0,sizeof(LayerInfo));
//        inpar.Batch              = l->Info.Batch;
//        //inpar.Steps            = l->Info.Steps;
//       // inpar.Image.Height       = l->Info.Image.Height;
//       // inpar.Image.Width        = l->Info.Image.Width;
//       // inpar.Image.Channels     = l->Info.Image.Channels;
//        inpar.Filters            = l->Info.Filters;
//        inpar.FilterSize         = blur_size;
//        inpar.StrideX            = blur_stride_x;
//        inpar.StrideY            = blur_stride_y;
//        inpar.Dilation           = 1;
//        inpar.Padding            = blur_pad;
//        inpar.Activation         = ACTIVATION_LINEAR;
//        inpar.BatchNormalize     = false;
//        inpar.Optimizer          = OPTIMIZER_NONE;
//        inpar.Index              = l->Info.Index;
//        inpar.Antialiasing       = false;
//        inpar.AssistedExcitation = false;
//        inpar.Deform             = false;
//        inpar.IsTrainMode        = l->Info.IsTrainMode;
//
//        l->InputLayer            = make_convolutional_layer(&inpar, 0);
//
//
//        const int blur_nweights = l->Info.Filters * blur_size * blur_size;  // (n / n) * n * blur_size * blur_size;
//
//        if (blur_size == 2)
//        {
//            for (i = 0; i < blur_nweights; i += (blur_size * blur_size)) 
//            {
//                l->InputLayer->Weights->Data[i + 0] = 1 / 4.f;
//                l->InputLayer->Weights->Data[i + 1] = 1 / 4.f;
//                l->InputLayer->Weights->Data[i + 2] = 1 / 4.f;
//                l->InputLayer->Weights->Data[i + 3] = 1 / 4.f;
//            }
//        }
//        else
//        {
//            for (i = 0; i < blur_nweights; i += (blur_size * blur_size)) 
//            {
//                l->InputLayer->Weights->Data[i + 0] = 1 / 16.f;
//                l->InputLayer->Weights->Data[i + 1] = 2 / 16.f;
//                l->InputLayer->Weights->Data[i + 2] = 1 / 16.f;
//                l->InputLayer->Weights->Data[i + 3] = 2 / 16.f;
//                l->InputLayer->Weights->Data[i + 4] = 4 / 16.f;
//                l->InputLayer->Weights->Data[i + 5] = 2 / 16.f;
//                l->InputLayer->Weights->Data[i + 6] = 1 / 16.f;
//                l->InputLayer->Weights->Data[i + 7] = 2 / 16.f;
//                l->InputLayer->Weights->Data[i + 8] = 1 / 16.f;
//            }
//        }
//    }
//    return l;
//}

void test_convolutional_layer()
{
    /*ConvolutionalLayer l = make_convolutional_layer(1, 1, 5, 5, 3, 2, 1, 5, 2, 2, 1, 1, ACTIVATION_LEAKY, 1, 0, 0, 0, 0, 0, 0, NULL, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = { 1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3 };
    network_state state = { 0 };
    state.input = data;
    forward_convolutional_layer(l, state);*/
}

//
//void assisted_excitation_forward(Layer* l, NetworkState state)
//{
////    const int iteration_num = (*state.Net.Seen) / (state.Net.Batch * state.Net.Subdivisions);
////
////    // epoch
////    //const float epoch = (float)(*state.net.seen) / state.net.train_images_num;
////
////    // calculate alpha
////    //const float alpha = (1 + cos(3.141592 * iteration_num)) / (2 * state.net.max_batches);
////    //const float alpha = (1 + cos(3.141592 * epoch)) / (2 * state.net.max_batches);
////    float alpha = (1 + cos(3.141592 * iteration_num / state.Net.MaxBatches));
////
////    if (l->Info.AssistedExcitation > 1)
////    {
////        if (iteration_num > l->Info.AssistedExcitation) alpha = 0;
////                                                   else alpha = (1 + cos(3.141592 * iteration_num / l->Info.AssistedExcitation));
////    }
////
////    //printf("\n epoch = %f, alpha = %f, seen = %d, max_batches = %d, train_images_num = %d \n",
////    //    epoch, alpha, (*state.net.seen), state.net.max_batches, state.net.train_images_num);
////
////      // TO-DO :
////    //int whb = l.out_w * l.out_h * l.batch * sizeof(float);
////    uint whbs = l->Output.TotalSize * (uint)sizeof(float);
////
////    float* a_avg = (float*)malloc(whbs);
////    float* g     = (float*)malloc(whbs);
////
////    int b;
////    int w, h, c;
////    int pos = 0;
////
////    l->Info.MaxBoxes = state.Net.NumBoxes;
////    l->Info.Truths   = l->Info.MaxBoxes * (4 + 1);
////
////    for (b = 0; b < l->Info.Batch; ++b)
////    {
////        // calculate G
////        int t;
////        for (t = 0; t < state.Net.NumBoxes; ++t)
////        {
////            Box truth = float_to_box_stride(state.Net.Truth + t * (4 + 1) + b * l->Info.Truths, 1);
////            if (!truth.x) break;  // continue;
////
////            int left   = floor((truth.x - truth.w / 2) * l->Output.Sizes[2]);//  l.out_w
////            int right  = ceil((truth.x + truth.w / 2)  * l->Output.Sizes[2]);//  l.out_w
////            int top    = floor((truth.y - truth.h / 2) * l->Output.Sizes[2]);//  l.out_w
////            int bottom = ceil((truth.y + truth.h / 2)  * l->Output.Sizes[1]);//  l.out_h
////
////            for (w = left; w <= right; w++) 
////            {
////                for (h = top; h < bottom; h++) 
////                {
////                    pos = w + l->Output.Sizes[2] * h + l->Output.Sizes[2] * l->Output.Sizes[1] * b;
////                    g[pos] = 1;
////                }
////            }
////        }
////    }
////
////    int p1 = 0, p2 = 0, p3 = 0;
////
////    for (b = 0; b < l->Info.Batch; ++b)
////    {
////        // calculate average A
////        for (w = 0; w < l->Output.Sizes[2]; w++)
////        {
////            for (h = 0; h < l->Output.Sizes[1]; h++)
////            {
////                for (c = 0; c < l->Output.Sizes[0]; c++)
////                {
////                    p1 = w + l->Output.Sizes[2] * (h + l->Output.Sizes[1] * b);
////                    p2 = w + l->Output.Sizes[2] * (h + l->Output.Sizes[1] * (c + l->Output.Sizes[0] * b));
////                    a_avg[p1] += l->Output.Data[p2];
////                }
////
////                p3 = w + l->Output.Sizes[2] * (h + l->Output.Sizes[1] * b);
////                a_avg[p3] /= l->Output.Sizes[0];  // a_avg / d
////            }
////        }
////    }
////
////    // change activation
////    for (b = 0; b < l->Info.Batch; ++b)
////    {
////        for (w = 0; w < l->Output.Sizes[2]; w++)
////        {
////            for (h = 0; h < l->Output.Sizes[1]; h++)
////            {
////                for (c = 0; c < l->Output.Sizes[0]; c++)
////                {
////                    // a = a + alpha(t) + e(c,i,j) = a + alpha(t) + g(i,j) * avg_a(i,j) / channels
////
////                    p1 = w + l->Output.Sizes[2] * (h + l->Output.Sizes[1] * (c + l->Output.Sizes[0] * b));
////                    p2 = w + l->Output.Sizes[2] * (h + l->Output.Sizes[1] * b);
////                    p3 = w + l->Output.Sizes[2] * (h + l->Output.Sizes[1] * b);
////
////                    l->Output.Data[p1] += alpha * g[p2] * a_avg[p3];
////
////                    //l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))] =
////                    //    alpha * g[w + l.out_w*(h + l.out_h*b)] * a_avg[w + l.out_w*(h + l.out_h*b)];
////                }
////            }
////        }
////    }
////
//////    if (0)   // visualize ground truth
//////    {
//////#ifdef OPENCV
//////        for (b = 0; b < l.batch; ++b)
//////        {
//////            image img = float_to_image(l.out_w, l.out_h, 1, &g[l.out_w * l.out_h * b]);
//////            char buff[100];
//////            sprintf(buff, "a_excitation_%d", b);
//////            show_image_cv(img, buff);
//////
//////            image img2 = float_to_image(l.out_w, l.out_h, 1, &l.output[l.out_w * l.out_h * l.out_c * b]);
//////            char buff2[100];
//////            sprintf(buff2, "a_excitation_act_%d", b);
//////            show_image_cv(img2, buff2);
//////            wait_key_cv(5);
//////        }
//////        wait_until_press_key_cv();
//////#endif // OPENCV
//////    }
////
////    free(g);
////    free(a_avg);
//}
//




//
//void _forward_convolutional_layer(ConvolutionalLayer *l, NetworkState state)
//{
//    //int out_h = convolutional_out_height(l);
//    //int out_w = convolutional_out_width(l);
//    int i, j;
//
//    // TO-DO :
//    //fill_cpu(l.outputs * l.batch, 0, l.output, 1);
//
//    int   m       = l->Info.Filters / l->Info.Groups;
//    int   k       = l->Info.FilterSize * l->Info.FilterSize * l->Info.Image.Channels / l->Info.Groups;
//    int   n       = out_h * out_w;
//    float ch_g    = l->Info.Image.Channels / l->Info.Groups;
//    float pre_pos = ch_g * l->Info.Image.Height * l->Info.Image.Width;
//
//    static int u = 0;
//    u++;
//
//    for (i = 0; i < l->Info.Batch; ++i)
//    {
//        for (j = 0; j < l->Info.Groups; ++j)
//        {
//            float* a = l->Weights->Data + j * l->Weights->Size / l->Info.Groups;
//            float* b = state.Workspace.Data;
//            float* c = l->Output.Data + (i * l->Info.Groups + j) * n * m;
//
//            //printf(" l.index = %d - FP32 \n", l.index);
//            // TO-DO : ??? erro acesso
//            //float* im = state.Input.Data + (float)(i * l->Info.Groups + j) * pre_pos;
//            float* im = 0;
//
//            if (l->Info.FilterSize == 1 && l->Info.Stride == 1 && l->Info.Dilation == 1) 
//            {
//                b = im;
//            }
//            else 
//            {
//                //im2col_cpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
//                im2col_cpu_ext
//                (
//                    im,                                                                     // input
//                    ch_g,                                                                   // input channels
//                    l->Info.Image.Height, l->Info.Image.Width,                              // input size (h, w)
//                    l->Info.FilterSize, l->Info.FilterSize,                                 // kernel size (h, w)
//                    l->Info.Padding * l->Info.Dilation, l->Info.Padding * l->Info.Dilation, // padding (h, w)
//                    l->Info.StrideY, l->Info.StrideX,                                       // stride (h, w)
//                    l->Info.Dilation, l->Info.Dilation,                                     // dilation (h, w)
//                    b                                                                       // output
//                );             
//            }
//
//            general_matrix_multiplier(false, false, m, n, k, 1, a, k, b, n, 1, c, n);
//            //c += n*m;
//            //state.input += l.c*l.h*l.w;
//        }
//    }
//
//    if (l->Info.BatchNormalize)
//    {
//        // TO-DO : implemetar em batchnorm_layer.c
//        //forward_batchnorm_layer(l, state);
//    }
//    else
//    {
//        add_bias(l->Output.Data, l->Biases->Data, l->Info.Batch, l->Info.Filters, out_h * out_w);
//    }
//
//    int tb = l->Output.TotalSize * l->Info.Batch;
//    int hw = l->Output.Sizes[2] * l->Output.Sizes[1];
//    int ch = l->Output.Sizes[0];
//
//    // TO-DO: linha 383 'l->Output.Dimensions, l->Output.Sizes[0]', no original usa parametros do RGB da imagem, nao Tensor. analizar melhor a substituicao...
//
//    if (l->Info.Activation == ACTIVATION_SWISH)
//    {
//        activate_array_swish(l->Output.Data, tb, l->ActivationInput.Data, l->Output.Data);
//    } 
//    else if (l->Info.Activation == ACTIVATION_MISH)
//    {
//        activate_array_mish(l->Output.Data, tb, l->ActivationInput.Data, l->Output.Data);
//    }
//    else if (l->Info.Activation == ACTIVATION_HARD_MISH)
//    {
//        activate_array_hard_mish(l->Output.Data, tb, l->ActivationInput.Data, l->Output.Data);
//    }
//    else if (l->Info.Activation == ACTIVATION_NORM_CHAN)
//    {
//        activate_array_normalize_channels(l->Output.Data, tb, l->Info.Batch, ch, hw, l->Output.Data);
//    }
//    else if (l->Info.Activation == ACTIVATION_NORM_CHAN_SOFTMAX)
//    {
//        activate_array_normalize_channels_softmax(l->Output.Data, tb, l->Info.Batch, ch, hw, l->Output.Data, 0);
//    }
//    else if (l->Info.Activation == ACTIVATION_NORM_CHAN_SOFTMAX_MAXVAL)
//    {
//        activate_array_normalize_channels_softmax(l->Output.Data, tb, l->Info.Batch, ch, hw, l->Output.Data, 1);
//    }
//    else
//    {
//        activate_array_cpu_custom(l->Output.Data, tb, l->Info.Activation);
//    }
//
//
//    if (l->Info.AssistedExcitation && state.IsTrain)
//    {
//        assisted_excitation_forward(l, state);
//    }
//
//     // TO-DO : implemetar...
//    //if (l.antialiasing) {
//    //    network_state s = { 0 };
//    //    s.train = state.train;
//    //    s.workspace = state.workspace;
//    //    s.net = state.net;
//    //    s.input = l.output;
//    //    forward_convolutional_layer(*(l.input_layer), s);
//    //    //simple_copy_ongpu(l.outputs*l.batch, l.output, l.input_antialiasing);
//    //    memcpy(l.output, l.input_layer->output, l.input_layer->outputs * l.input_layer->batch * sizeof(float));
//    //}
//}


//
//void backward_convolutional_layer(ConvolutionalLayer l, NetworkState state)
//{
//    //int i, j;
//    //int m  = l.Info.Filters / l.Info.Groups;
//    //int n  = l.Info.FilterSize * l.Info.FilterSize * l.Info.Image.Channels / l.Info.Groups;
//    //int k  = l.Output.Sizes[2] * l.Output.Sizes[1];
//    //int ob = l.Output.TotalSize * l.Info.Batch;
//    //int ch = l.Output.Sizes[0];
//    //int h  = l.Output.Sizes[0];
//    //int w  = l.Output.Sizes[0];
//
//    //if (l.Info.Activation == ACTIVATION_SWISH)
//    //{
//    //    gradient_array_swish(l.Output.Data, ob, l.ActivationInput.Data, l.TraiInfo.Delta.Data);
//    //}
//    //else if (l.Info.Activation == ACTIVATION_MISH)
//    //{
//    //    gradient_array_mish(ob, l.ActivationInput.Data, l.TraiInfo.Delta.Data);
//    //}
//    //else if (l.Info.Activation == ACTIVATION_HARD_MISH)
//    //{
//    //    gradient_array_hard_mish(ob, l.ActivationInput.Data, l.TraiInfo.Delta.Data);
//    //}
//    //else if (l.Info.Activation == ACTIVATION_NORM_CHAN_SOFTMAX || l.Info.Activation == ACTIVATION_NORM_CHAN_SOFTMAX_MAXVAL)
//    //{
//    //    gradient_array_normalize_channels_softmax(l.Output.Data, ob, l.Info.Batch, ch, w * h, l.TraiInfo.Delta.Data);
//    //}
//    //else if (l.Info.Activation == ACTIVATION_NORM_CHAN)
//    //{
//    //    gradient_array_normalize_channels(l.Output.Data, ob, l.Info.Batch, ch, w * h, l.TraiInfo.Delta.Data);
//    //}
//    //else
//    //{
//    //    gradient_array(l.Output.Data, ob, l.Info.Activation, l.TraiInfo.Delta.Data);
//    //}
//
//    //// TO-DO : implemetar...
//    //if (l.Info.BatchNormalize) 
//    //{
//    //    //backward_batchnorm_layer(l, state);
//    //}
//    //else 
//    //{
//    //    //backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
//    //}
//
//    //for (i = 0; i < l.Info.Batch; ++i) 
//    //{
//    //    for (j = 0; j < l.Info.Groups; ++j)
//    //    {
//    //        float* a = l.TraiInfo.Delta.Data + (i * l.Info.Groups + j) * m * k;
//    //        float* b = state.Workspace.Data;
//    //        float* c = l.WeightUpdates->Data + j * l.Weights->Size / l.Info.Groups;
//
//    //        float* im = state.Input.Data + (i * l.Info.Groups + j) * (ch / l.Info.Groups) * h * w;
//
//    //        //im2col_cpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
//    //        im2col_cpu_ext
//    //        (
//    //            im,                 // input
//    //            ch / l.Info.Groups,     // input channels
//    //            h, w,           // input size (h, w)
//    //            l.Info.FilterSize, l.Info.FilterSize,     // kernel size (h, w)
//    //            l.Info.Padding * l.Info.Dilation, l.Info.Padding * l.Info.Dilation,       // padding (h, w)
//    //            l.Info.StrideY, l.Info.StrideX, // stride (h, w)
//    //            l.Info.Dilation, l.Info.Dilation, // dilation (h, w)
//    //            b// output
//    //        );                 
//
//    //        general_matrix_multiplier(false, true, m, n, k, 1, a, k, b, k, 1, c, n);
//
//    //        if (state.Delta.Size > 0) 
//    //        {
//    //            a = l.Weights->Data + j * l.Weights->Size / l.Info.Groups;
//    //            b = l.TraiInfo.Delta.Data + (i * l.Info.Groups + j) * m * k;
//    //            c = state.Workspace.Data;
//
//    //            general_matrix_multiplier(true, false, n, k, m, 1, a, n, b, k, 0, c, k);
//
//    //            //col2im_cpu(state.workspace, l.c / l.groups, l.h, l.w, l.size, l.stride,
//    //            //     l.pad, state.delta + (i*l.groups + j)*l.c / l.groups*l.h*l.w);
//
//    //            col2im_cpu_ext
//    //            (
//    //                state.Workspace.Data,        // input
//    //                ch / l.Info.Groups,         // input channels (h, w)
//    //                h, w,               // input size (h, w)
//    //                l.Info.FilterSize, l.Info.FilterSize,         // kernel size (h, w)
//    //                l.Info.Padding * l.Info.Dilation, l.Info.Padding * l.Info.Dilation,           // padding (h, w)
//    //                l.Info.StrideY, l.Info.StrideX,     // stride (h, w)
//    //                l.Info.Dilation, l.Info.Dilation, // dilation (h, w)
//    //                l.TraiInfo.Delta.Data + (i * l.Info.Groups + j) * (ch / l.Info.Groups) * h * w// output (delta)
//    //            ); 
//    //        }
//    //    }
//    //}
//}
//
//
//
//
//void update_convolutional_layer(ConvolutionalLayer l, int batch, float learning_rate_init, float momentum, float decay)
//{
//    float learning_rate = learning_rate_init * l.Info.LearningRateScale;
//    //float momentum = a.momentum;
//    //float decay = a.decay;
//    //int batch = a.batch;
//
//    axpy_cpu(l.Weights->TotalSize, -decay * batch, l.Weights->Data, 1, l.TraiInfo.WeightUpdates->Data, 1);
//    axpy_cpu(l.Weights->TotalSize, learning_rate / batch, l.TraiInfo.WeightUpdates->Data, 1, l.Weights->Data, 1);
//    scal_cpu(l.Weights->TotalSize, momentum, l.TraiInfo.WeightUpdates->Data, 1);
//
//    axpy_cpu(l.Info.Filters, learning_rate / batch, l.TraiInfo.BiasUpdates->Data, 1, l.Biases->Data, 1);
//    scal_cpu(l.Info.Filters , momentum, l.TraiInfo.BiasUpdates->Data, 1);
//
//    if (l.BatchNormalize.Scales)
//    {
//        axpy_cpu(l.Info.Filters, learning_rate / batch, l.BatchNormalize.ScaleUpdates->Data, 1, l.BatchNormalize.Scales->Data, 1);
//        scal_cpu(l.Info.Filters, momentum, l.BatchNormalize.ScaleUpdates->Data, 1);
//    }
//}
//
//
//void forward_convolutional_layer_gpu(ConvolutionalLayer l, NetworkState state)
//{
//
//}
//
//void backward_convolutional_layer_gpu(ConvolutionalLayer l, NetworkState state)
//{
//
//}
//
//
//void update_convolutional_layer_gpu(ConvolutionalLayer l, int batch, float learning_rate_init, float momentum, float decay)
//{
//
//}

