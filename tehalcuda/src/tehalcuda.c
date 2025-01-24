#include "../include/tehalcuda.h"
#include "eventhub.h"
#include "allocator.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <string.h>



BufferFloat* tehal_create_buffer(bool is_device, uint size, bool set_value, float value)
{
	BufferFloat* buffer = (BufferFloat*)mem_allocate(sizeof(BufferFloat));

	tehal_init_buffer(buffer, is_device, size, set_value, value);
	return buffer;
}

void tehal_init_buffer(BufferFloat* buffer, bool is_device, uint size, bool set_value, float value)
{
	buffer->Size     = 0;
	buffer->MaxSize  = size;
	buffer->IsDevice = is_device;
	buffer->Data     = tehal_allocate(size * sizeof(float));

	if (set_value)
	{
		float* ar = malloc(size * sizeof(float));
		int ix = 0;
		while (ix < buffer->MaxSize)
		{
			ar[ix] = value;
			ix++;
		}
		tehal_copy_memo(ar, buffer->Data, buffer->MaxSize, buffer->MaxSize);
		free(ar);
	}
}


TensorFloat* tehal_create_tensor(bool is_device, int dim, int* sizes)
{
	TensorFloat* tensor = 0;
	tensor = (TensorFloat*)mem_allocate(sizeof(TensorFloat));

	tehal_init_tensor(tensor, is_device, dim, sizes);
	return tensor;
}

void tehal_init_tensor(TensorFloat* tensor, bool is_device, int dim, int* sizes)
{
	tensor->Dimensions = dim;
	tensor->IsDevice   = is_device;
	tensor->TotalSize  = 0;
	tensor->Sizes      = tehal_allocate(dim * sizeof(int));

	int ic = 0;
	while (ic < dim)
	{
		tensor->Sizes[ic]  = sizes[ic];
		tensor->TotalSize += sizes[ic];
		ic++;
	}

	tensor->Data = tehal_allocate(tensor->TotalSize);
	tensor->Grad = tehal_allocate(tensor->TotalSize);

	return tensor;
}

void tehal_release_tensor(TensorFloat* tensor)
{
	tehal_deallocate(tensor->Data);
	tehal_deallocate(tensor->Grad);
	tensor->Dimensions = 0;
}

void tehal_free_tensor(TensorFloat* tensor)
{
	tehal_release_tensor(tensor);
	mem_deallocate(tensor->Sizes);
	mem_deallocate(tensor);
}

void tehal_copy_from_device(TensorFloat* source, TensorFloat* dest)
{
	cudaError_t status = cudaMemcpy(dest->Data, source->Data, dest->TotalSize * sizeof(float), cudaMemcpyDeviceToHost);

	if (status != cudaSuccess)
	{
		char* msg = cudaGetErrorString(status);
		error(status, msg);
		return 0;
	}
}

void tehal_copy_to_device(TensorFloat* source, TensorFloat* dest)
{
	cudaError_t status = cudaMemcpy(dest->Data, source->Data, dest->TotalSize * sizeof(float), cudaMemcpyHostToDevice);

	if (status != cudaSuccess)
	{
		char* msg = cudaGetErrorString(status);
		error(status, msg);
		return 0;
	}
}

void tehal_copy_memo(float* source, float* dest, uint source_size, uint dest_size)
{
	if (source_size != dest_size)
	{
		error(0, "tehal_copy_memo(). origem e destino com tamanho da memoria diferente.");
		return;
	}

	cudaError_t status = cudaMemcpy(dest, source, source_size * sizeof(float), cudaMemcpyDeviceToDevice);

	if (status != cudaSuccess)
	{
		char* msg = cudaGetErrorString(status);
		error(status, msg);
		return 0;
	}
}

void tehal_copy(TensorFloat* source, TensorFloat* dest)
{
	if (source->TotalSize != dest->TotalSize)
	{
		error(0, "tehal_copy(). origem e destino com tamanho da memoria diferente.");
		return;
	}

	cudaError_t status = cudaMemcpy(dest->Data, source->Data, dest->TotalSize * sizeof(float), cudaMemcpyDeviceToDevice);

	if (status != cudaSuccess)
	{
		char* msg = cudaGetErrorString(status);
		error(status, msg);
		return 0;
	}
}


void* tehal_allocate(uint size)
{
	void* memo = 0;

	cudaError_t status = cudaMalloc((void**)&memo, size);

	if (status != cudaSuccess)
	{
		char* msg = cudaGetErrorString(status);
		error(status,msg);
		return 0;
	}

	return memo;
}

void tehal_deallocate(void* arg)
{
	cudaError_t status = cudaFree(arg);

	if (status != cudaSuccess)
	{
		char* msg = cudaGetErrorString(status);
		error(status, msg);
		return 0;
	}
}
