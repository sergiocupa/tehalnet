#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "activation.h"
    #include "../../include/tehalnet.h"



	typedef Layer ActivationLayer;


	//ActivationLayer* make_activation_layer(LayerInfo* param, ActivationLayer* share_layer);
	ActivationLayer* make_activation_layer(TensorFloat* template_tensor, ActivationOption activation);


#ifdef __cplusplus
}
#endif

#endif /* ACTIVATION_LAYER */