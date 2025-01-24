#include "activation_layer.h"
#include "../util/convolutional_util.h"
#include <string.h>


void forward_activation_layer(Layer* layer, TensorFloat* input, TensorFloat* output)
{
    if (!input->IsStreammed)
    {
        tensor_to_stream(input);
        input->IsStreammed = true;
    }

    activate_tensor(input, output, layer->Info.Activation);

    if (layer->BuildOutputTensor)
    {
        stream_to_tensor(output);
    }
}

void backward_activation_layer(ActivationLayer* l, NetworkState* state)
{
    gradient_tensor(l->Input, l->Info.Activation, l->TraiInfo.Delta);
    TehLib.copy(l->TraiInfo.Delta, state->Delta);
}


ActivationLayer* make_activation_layer(TensorFloat* template_tensor, ActivationOption activation)
{
    ActivationLayer* l = (ActivationLayer*)TehLib.allocate(false, sizeof(ActivationLayer));
    memset(l, 0, sizeof(ActivationLayer));

    l->Info.Type       = LAYER_ACTIVE;
    l->Info.Batch      = template_tensor->Sizes[0];
    l->Info.Activation = activation;
    l->Input           = TehLib.create_tensor(template_tensor->IsDevice, template_tensor->Dimensions, template_tensor->Sizes);
    l->Output          = TehLib.create_tensor(template_tensor->IsDevice, template_tensor->Dimensions, template_tensor->Sizes);
    l->TraiInfo.Delta  = TehLib.create_tensor(template_tensor->IsDevice, template_tensor->Dimensions, template_tensor->Sizes);

    if (template_tensor->IsDevice)
    {

    }
    else
    {
        l->Forward  = forward_activation_layer;
        l->Backward = backward_activation_layer;
    }
    return l;
}
