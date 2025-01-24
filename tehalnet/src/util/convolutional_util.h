#ifndef CONVOLUTIONAL_UTIL_H
#define CONVOLUTIONAL_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "../../include/tehalnet.h"



	void tensor_to_stream(TensorFloat* tensor);
	void stream_to_tensor(TensorFloat* tensor);



#ifdef __cplusplus
}
#endif

#endif /* CONVOLUTIONAL_UTIL */