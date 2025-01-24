#include "../../include/tehalnet.h"
#include "../util/convolutional_util.h"
#include "../util/blas.h"
#include <string.h>


typedef enum BathNormalizeElement
{
    BATCH_NORMALIZATION_ELEMENT_SCALES           = 0,
    BATCH_NORMALIZATION_ELEMENT_ROLLING_MEAN     = 1,
    BATCH_NORMALIZATION_ELEMENT_ROLLING_VARIANCE = 2,
    BATCH_NORMALIZATION_ELEMENT_SCALES_EMA       = 3,
    BATCH_NORMALIZATION_ELEMENT_SCALE_UPDATES    = 4,
    BATCH_NORMALIZATION_ELEMENT_MEAN             = 5,
    BATCH_NORMALIZATION_ELEMENT_VARIANCE         = 6,
    BATCH_NORMALIZATION_ELEMENT_MEAN_DELTA       = 7,
    BATCH_NORMALIZATION_ELEMENT_VARIANCE_DELTA   = 8,
    BATCH_NORMALIZATION_ELEMENT_BIASES           = 9,
    BATCH_NORMALIZATION_ELEMENT_BIAS_UPDATES     = 10
};



Layer* make_batch_normalize_layer(TensorInfo input, int output_filters, bool is_train_mode)
{
    Layer* l = (Layer*)TehLib.allocate(false, sizeof(Layer));
    memset(l, 0, sizeof(Layer));

    l->Info.Type             = LAYER_BATCHNORM;
    l->Info.Batch            = input.Sizes[0];
    l->Info.IsTrainMode      = is_train_mode;
    l->Info.Filters          = output_filters;
    l->Info.Input.IsDevice   = input.IsDevice;
    l->Info.Input.Dimensions = input.Dimensions;
    l->Info.Input.Sizes      = (int*)TehLib.allocate(false, (sizeof(int)* input.Dimensions));

    TehLib.copy_int(input.Sizes, l->Info.Input.Sizes, input.Dimensions, l->Info.Input.Dimensions, false, false);

    init_tensor_array(&l->Elements);
    tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, (int[]) { l->Info.Filters }), BATCH_NORMALIZATION_ELEMENT_SCALES);
    tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, (int[]) { l->Info.Filters }), BATCH_NORMALIZATION_ELEMENT_ROLLING_MEAN);
    tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, (int[]) { l->Info.Filters }), BATCH_NORMALIZATION_ELEMENT_ROLLING_VARIANCE);
    tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, (int[]) { l->Info.Filters }), BATCH_NORMALIZATION_ELEMENT_BIASES);
    tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, (int[]) { l->Info.Filters }), BATCH_NORMALIZATION_ELEMENT_BIAS_UPDATES);


    if (is_train_mode)
    {
        tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, (int[]) { l->Info.Filters }), BATCH_NORMALIZATION_ELEMENT_SCALES_EMA);
        tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, (int[]) { l->Info.Filters }), BATCH_NORMALIZATION_ELEMENT_SCALE_UPDATES);
        tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, (int[]) { l->Info.Filters }), BATCH_NORMALIZATION_ELEMENT_MEAN);
        tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, (int[]) { l->Info.Filters }), BATCH_NORMALIZATION_ELEMENT_VARIANCE);
        tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, (int[]) { l->Info.Filters }), BATCH_NORMALIZATION_ELEMENT_MEAN_DELTA);
        tensor_array_add_index(&l->Elements, TehLib.create_tensor(input.IsDevice, 1, (int[]) { l->Info.Filters }), BATCH_NORMALIZATION_ELEMENT_VARIANCE_DELTA);
    }
}



void PrepareStream(TensorFloat* input, LayerInfo* info)
{
    if (!input->Stream->Dimensions)
    {
        int batch   = input->Sizes[0];
        int in_h    = input->Sizes[1];
        int in_w    = input->Sizes[2];
        int filters = input->Sizes[3];

        int group_in_ch = filters / info->Groups;
        int k = info->KernelSize;
        int p = info->Padding;
        int s = info->Stride;
        int d = info->KernelDilation;

        im2col_cpu_ext(input->Data, group_in_ch, in_h, in_w, k, k, p, p, s, s, d, d, input->Stream->Data);
    }
}


void forward_batch_normalize_layer(Layer* layer, TensorFloat *input, TensorFloat* output)
{
    if (!input->IsStreammed)
    {
        tensor_to_stream(input);// &layer->Info);
    }

    int sz = output->Sizes[1] * output->Sizes[2];

    if (layer->TraiInfo.IsEnabled)
    {
        //mean_cpu(l.output, l.batch, l.out_c, l.out_h * l.out_w, l.mean);
        //variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h * l.out_w, l.variance);

        //scal_cpu(l.out_c, .9, l.rolling_mean, 1);
        //axpy_cpu(l.out_c, .1, l.mean, 1, l.rolling_mean, 1);
        //scal_cpu(l.out_c, .9, l.rolling_variance, 1);
        //axpy_cpu(l.out_c, .1, l.variance, 1, l.rolling_variance, 1);

        //copy_cpu(l.outputs * l.batch, l.output, 1, l.x, 1);
        //normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h * l.out_w);
        //copy_cpu(l.outputs * l.batch, l.output, 1, l.x_norm, 1);
    }
    else
    {
        TensorFloat* r_mean     = layer->Elements.Data[BATCH_NORMALIZATION_ELEMENT_ROLLING_MEAN];
        TensorFloat* r_variance = layer->Elements.Data[BATCH_NORMALIZATION_ELEMENT_ROLLING_VARIANCE];

        normalize_cpu(input->Stream->Data, r_mean->Data, r_variance->Data, layer->Info.Batch, layer->Info.Filters, sz, output->Stream->Data);
    }

    TensorFloat* scales = layer->Elements.Data[BATCH_NORMALIZATION_ELEMENT_SCALES];
    TensorFloat* biases = layer->Elements.Data[BATCH_NORMALIZATION_ELEMENT_BIASES];
    TensorFloat* biasup = layer->Elements.Data[BATCH_NORMALIZATION_ELEMENT_BIAS_UPDATES];

    scale_bias(output->Stream->Data, scales, layer->Info.Batch, layer->Info.Filters, sz);
    add_bias(output->Stream->Data, biases, layer->Info.Batch, layer->Info.Filters, sz);


    if (layer->BuildOutputTensor)
    {
        stream_to_tensor(output);
    }
}