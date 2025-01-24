#ifndef LOCAL_LAYER_H
#define LOCAL_LAYER_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "../../include/tehalnet.h"


    typedef Layer LocalLayer;

    LocalLayer* make_local_layer(LayerInfo* param);



#ifdef __cplusplus
}
#endif

#endif /* LOCAL_LAYER */