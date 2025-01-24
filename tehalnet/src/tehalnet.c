//#include "../include/tehalnet.h"
//#include "layer/convolutional_layer.h"
//#include <string.h>
//
//void init_default_layer(LayerInfo* layer)
//{
//    memset(&layer, 0, sizeof(LayerInfo));
//
//    int height   = 600;
//    int width    = 800;
//    int channels = 3;
//
//    layer->Input.IsDevice    = false;
//    layer->Batch             = 1;
//    layer->Input.Dimensions  = 4;
//    layer->Input.Sizes       = (int[]){ layer->Batch, height, width, channels };
//    layer->Input.TotalSize   = layer->Input.Sizes[0] + layer->Input.Sizes[1] + layer->Input.Sizes[2] + layer->Input.Sizes[3];
//    layer->Output.IsDevice   = false;
//    //layer->Output.Dimensions = 4;
//    //layer->Output.Sizes      = (int[]){ 1, 800, 600, 3 };
//    //layer->Output.TotalSize  = layer->Input.Sizes[0] + layer->Input.Sizes[1] + layer->Input.Sizes[2] + layer->Input.Sizes[3];
//    layer->Activation        = ACTIVATION_LINEAR;
//    layer->Groups            = 1;
//    layer->IsBatchNormalize  = 1;
//    layer->Type              = LAYER_NONE;
//}
//
//
//void init_tensor_info(TensorInfo* reference, TensorInfo* info)
//{
//    info->IsDevice   = reference->IsDevice;
//    info->Dimensions = reference->Dimensions;
//    info->Sizes      = TehLib.allocate(false, reference->Dimensions * sizeof(int));// = param->Input.Sizes;
//
//    TehLib.copy_int(reference->Sizes, info->Sizes, reference->Dimensions, reference->Dimensions, false, false);
//}
//
