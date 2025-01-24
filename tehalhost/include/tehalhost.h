#ifndef TEHALHOST_H
#define TEHALHOST_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "../../tehaltype/include/tehaltype.h"
    #include "platform.h"


	PLATFORM_API void         tehal_copy_int(int* source, int* dest, uint source_size, uint dest_size);
	PLATFORM_API void         tehal_copy_float(float* source, float* dest, uint source_size, uint dest_size);
	PLATFORM_API void         tehal_copy(TensorFloat* source, TensorFloat* dest);
	PLATFORM_API void*        tehal_allocate(uint size);
	PLATFORM_API void         tehal_deallocate(void* arg);
	PLATFORM_API BufferFloat* tehal_create_buffer(bool is_device, uint size, bool set_value, float value);
	PLATFORM_API void         tehal_init_buffer(BufferFloat* buffer, bool is_device, uint size, bool set_value, float value);
	PLATFORM_API TensorFloat* tehal_create_tensor(bool is_device, int dim, int* sizes);
	PLATFORM_API void         tehal_init_tensor(TensorFloat* tensor, bool is_device, int dim, int* sizes);
	PLATFORM_API void         tehal_release_tensor(TensorFloat* tensor);
	PLATFORM_API void         tehal_free_tensor(TensorFloat* tensor);


#ifdef __cplusplus
}
#endif

#endif /* TEHALHOST */