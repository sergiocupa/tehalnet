#include "../include/tehalnet.h"
#include "../../tehalhost/include/tehalhost.h"
#include "yason.h"
#include "eventhub.h"
#include <windows.h>
#include <string.h>
#include <stdio.h>


bool PlugInitialized = false;

TensorFloat* (*create_buffer_device)    (bool is_device, uint size, bool set_value, float value);
TensorFloat* (*init_buffer_device)      (BufferFloat* buffer, bool is_device, uint size, bool set_value, float value);
TensorFloat* (*create_tensor_device)    (bool is_device, int dim, int* sizes);
void         (*init_tensor_device)      (TensorFloat* tensor, bool is_device, int dim, int* sizes);
void         (*release_tensor_device)   (TensorFloat* tensor);
void         (*free_tensor_device)      (TensorFloat* tensor);
void* (*allocate_device)         (uint size);
void         (*deallocate_device)       (void*);
void         (*copy_device_from_device) (TensorFloat* source, TensorFloat* dest);
void         (*copy_device_to_device)   (TensorFloat* source, TensorFloat* dest);
void         (*copy_device)             (TensorFloat* source, TensorFloat* dest);
void         (*copy_memo_device)        (float* source, float* dest, uint source_size, uint dest_size);


void* teh_load_library(char* library)
{
	char libn[260];

#ifdef PLATFORM_WIN
	const char* EXT = ".dll";
#else
	const char* EXT = ".so";
#endif

	int extl = strlen(EXT);
	int libl = strlen(library);

	int ix = 0;
	while (ix < libl && ix < 259)
	{
		libn[ix] = library[ix];
		ix++;
	}
	int im = 0;
	while (im < extl && ix < 259)
	{
		libn[ix] = EXT[im];
		ix++;
		im++;
	}
	libn[ix] = 0;

#ifdef PLATFORM_WIN
	return LoadLibrary(TEXT(libn));
#else
	return dlopen(libn, RTLD_LAZY);
#endif
}

void* teh_load_get_method(void* handle, const char* name)
{
#ifdef PLATFORM_WIN
	return GetProcAddress(handle, name);
#else
	return dlsym(handle, name);
#endif
}

//
//void init_tensor_array(TensorFloatArray* ar)
//{
//	ar->Count    = 0;
//	ar->MaxCount = 100;
//	ar->Data     = (TensorFloat**)tehal_allocate(ar->MaxCount * sizeof(TensorFloat*));
//}
//
//TensorFloatArray* create_tensor_array()
//{
//	TensorFloatArray* ar = (TensorFloatArray*)tehal_allocate(sizeof(TensorFloatArray));
//
//	init_tensor_array(ar);
//	return ar;
//}
//
//void tensor_array_add(TensorFloatArray* _this, TensorFloat* item)
//{
//	if (_this)
//	{
//		if (_this->Count >= _this->MaxCount)
//		{
//			_this->MaxCount = _this->Count * 2;
//			_this->Data     = (void**)realloc((TensorFloat**)_this->Data, _this->MaxCount * sizeof(TensorFloat*));
//		}
//
//		_this->Data[_this->Count] = item;
//		_this->Count++;
//	}
//}
//void tensor_array_add_index(TensorFloatArray* _this, TensorFloat* item, int index)
//{
//	if (_this)
//	{
//		if (_this->Count >= _this->MaxCount)
//		{
//			_this->MaxCount = _this->Count * 2;
//			_this->Data = (void**)realloc((TensorFloat**)_this->Data, _this->MaxCount * sizeof(TensorFloat*));
//		}
//
//		_this->Data[index] = item;
//		_this->Count++;
//	}
//}



BufferFloat* create_buffer_switch(bool is_device, uint size, bool set_value, float value)
{
	if (TehLib.UseDevice && is_device)
	{
		create_buffer_device(is_device, size, set_value, value);
	}
	else
	{
		tehal_create_buffer(is_device, size, set_value, value);
	}
}
void init_buffer_switch(BufferFloat* buffer, bool is_device, uint size, bool set_value, float value)
{
	if (TehLib.UseDevice && is_device)
	{
		init_buffer_device(buffer, is_device, size, set_value, value);
	}
	else
	{
		tehal_init_buffer(buffer, is_device, size, set_value, value);
	}
}


TensorFloat* create_tensor_switch(bool is_device, int dim, int* sizes)
{
	if (TehLib.UseDevice && is_device)
	{
		create_tensor_device(is_device, dim, sizes);
	}
	else
	{
		tehal_create_tensor(is_device, dim, sizes);
	}
}
void init_tensor_switch(TensorFloat* tensor, bool is_device, int dim, int* sizes)
{
	if (TehLib.UseDevice && is_device)
	{
		init_tensor_device(tensor, is_device, dim, sizes);
	}
	else
	{
		tehal_init_tensor(tensor, is_device, dim, sizes);
	}
}
void release_tensor_switch(TensorFloat* tensor)
{
	if (TehLib.UseDevice && tensor->IsDevice)
	{
		release_tensor_device(tensor);
	}
	else
	{
		tehal_release_tensor(tensor);
	}
}
void free_tensor_switch(TensorFloat* tensor)
{
	if (TehLib.UseDevice && tensor->IsDevice)
	{
		free_tensor_device(tensor);
	}
	else
	{
		tehal_free_tensor(tensor);
	}
}

void* allocate_switch(bool is_device, uint size)
{
	if (TehLib.UseDevice && is_device)
	{
		return allocate_device(size);
	}
	else
	{
		return tehal_allocate(size);
	}
}
void deallocate_switch(bool is_device, void* m)
{
	if (TehLib.UseDevice && is_device)
	{
		deallocate_device(m);
	}
	else
	{
		tehal_deallocate(m);
	}
}
void copy_switch(TensorFloat* source, TensorFloat* dest)
{
	if (source->IsDevice == true && dest->IsDevice == true)
	{
		copy_device(source, dest);
	}
	else if (source->IsDevice == false && dest->IsDevice == false)
	{
		tehal_copy(source->Data, dest->Data);
	}
	else if (source->IsDevice == true)
	{
		copy_device_from_device(source, dest);
	}
	else
	{
		copy_device_to_device(source, dest);
	}
}

void copy_float_switch(float* source, float* dest, uint source_size, uint dest_size, bool source_device, bool dest_device)
{
	if (source_device == true && dest_device == true)
	{
		copy_memo_device(source, dest, source_size, dest_size);
	}
	else if (source_device == false && dest_device == false)
	{
		tehal_copy_float(source, dest, source_size, dest_size);
	}
	else
	{
		error(0, "Modo misto 'device-host' não foi implementado");
	}
}

void copy_int_switch(int* source, int* dest, uint source_size, uint dest_size, bool source_device, bool dest_device)
{
	if (source_device == true && dest_device == true)
	{
		error(0, "Modo 'device' não foi implementado");
	}
	else if (source_device == false && dest_device == false)
	{
		tehal_copy_int(source, dest, source_size, dest_size);
	}
	else
	{
		error(0, "Modo misto 'device-host' não foi implementado");
	}
}

TensorFloat* tehal_create_tensor_imagem(bool is_device, int width, int height, int channels)
{
	return create_tensor_switch(is_device, 3, (int[]) { channels, height, width });
}

void tehal_init_tensor_imagem(TensorFloat* tensor, bool is_device, int width, int height, int channels)
{
	init_tensor_switch(tensor, is_device, 3, (int[]) { channels, height, width });
}


bool teh_plug_assemble(bool use_device, int device_index, char* devide_lib)
{
	if (PlugInitialized) return true;

	TehLib.UseDevice = use_device;
	TehLib.DeviceIndex = device_index;

	// apontar funções de alocacao e criação de tensors

	if (TehLib.UseDevice)
	{
		void* hLib = teh_load_library(devide_lib);

		if (hLib == NULL)
		{
			error(0, "Could not load library '%s'", devide_lib);
			return false;
		}

		create_tensor_device    = teh_load_get_method(hLib, "tehal_create_tensor");
		init_tensor_device      = teh_load_get_method(hLib, "tehal_init_tensor");
		release_tensor_device   = teh_load_get_method(hLib, "tehal_release_tensor");
		free_tensor_device      = teh_load_get_method(hLib, "tehal_free_tensor");
		allocate_device         = teh_load_get_method(hLib, "tehal_allocate");
		deallocate_device       = teh_load_get_method(hLib, "tehal_deallocate");
		init_buffer_device      = teh_load_get_method(hLib, "tehal_init_buffer");
		create_buffer_device    = teh_load_get_method(hLib, "tehal_create_buffer");
		copy_device             = teh_load_get_method(hLib, "tehal_copy");
		copy_device_from_device = teh_load_get_method(hLib, "tehal_copy_from_device");
		copy_device_to_device   = teh_load_get_method(hLib, "tehal_copy_to_device");
		copy_memo_device        = teh_load_get_method(hLib, "tehal_copy_memo");
	}

	TehLib.copy_int             = copy_int_switch;
	TehLib.copy_float           = copy_float_switch;
	TehLib.copy                 = copy_switch;
	TehLib.allocate             = tehal_allocate;
	TehLib.deallocate           = tehal_deallocate;
	TehLib.create_tensor        = create_tensor_switch;
	TehLib.init_tensor          = init_tensor_switch;
	TehLib.release_tensor       = release_tensor_switch;
	TehLib.free_tensor          = free_tensor_switch;
	TehLib.create_buffer        = create_buffer_switch;
	TehLib.init_buffer          = init_buffer_switch;
	TehLib.create_tensor_imagem = tehal_create_tensor_imagem;
	TehLib.init_tensor_imagem   = tehal_init_tensor_imagem;

	PlugInitialized = true;
}


