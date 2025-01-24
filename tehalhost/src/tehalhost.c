#include "../include/tehalhost.h"
#include "allocator.h"
#include "eventhub.h"
#include <stdlib.h>
#include <string.h>



void tehal_copy_int(int* source, int* dest, uint source_size, uint dest_size)
{
	if (source_size != dest_size)
	{
		error(0, "tehal_copy_memo(). origem e destino com tamanho da memoria diferente.");
		return;
	}
	
	mem_copy_int( source, dest, source_size);
}

void tehal_copy_float(float* source, float* dest, uint source_size, uint dest_size)
{
	if (source_size != dest_size)
	{
		error(0, "tehal_copy_memo(). origem e destino com tamanho da memoria diferente.");
		return;
	}

	mem_copy_float(source, dest, source_size);
}

void tehal_copy(TensorFloat* source, TensorFloat* dest)
{
	if (source->TotalSize != dest->TotalSize)
	{
		error(0, "tehal_copy(). origem e destino com tamanho da memoria diferente.");
		return;
	}

	mem_copy_float(source->Data, dest->Data, source->TotalSize);
}

BufferFloat* tehal_create_buffer(bool is_device, uint size, bool set_value, float value)
{
	BufferFloat* buffer = (BufferFloat*)tehal_allocate(sizeof(BufferFloat));

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
		mem_set_float(buffer->Data, buffer->MaxSize, value);
	}
}

TensorFloat* tehal_create_tensor(bool is_device, int dim, int* sizes)
{
	TensorFloat* tensor = (TensorFloat*)tehal_allocate(sizeof(TensorFloat));

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
	tehal_deallocate(tensor->Sizes);
	tehal_deallocate(tensor);
}



void* tehal_allocate(uint size)
{
	void* memo = mem_allocate(size);

	if (!memo)
	{
		error(0, "malloc error");
		return 0;
	}

	return memo;
}

void tehal_deallocate(void* arg)
{
	mem_deallocate(arg);
}
