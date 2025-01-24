#ifndef TEHALTYPE_H
#define TEHALTYPE_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "platform.h"


	typedef struct _TensorFloat      TensorFloat;
	typedef struct _TensorFloatArray TensorFloatArray;
	


	typedef struct _ImageFloat
	{
		int    Width;
		int    Height;
		int    Channels;
		int    Size;
		float* Data;
	}
	ImageFloat;

	typedef struct _ImageTransformInfo
	{
		int Rotate;
		float Reverse;
		float Angle;
		int Flipped;

	}
	ImageTransformInfo;

	typedef struct _BufferFloat
	{
		bool   IsDevice;
		int    MaxSize;
		int    Size;
		float* Data;
	}
	BufferFloat;

	typedef struct _TensorInfo
	{
		bool IsDevice;
		uint TotalSize;
		int  Dimensions;
		int* Sizes;
	}
	TensorInfo;



	struct _TensorFloat
	{
		// Item que ja foi espandido, transformado de 4D para 2D por exemplo. 
		// Campo usado para não precisar recalcular o tensor.
		// Entrada de proxima convolução não reaproveita o campo Stretch. 
		// Normalmente quem usa Stretch são funções auxiliares a convolução, por exemplo Activate() ou BatchNormalizate()
		// Antes de usar o Stretch, verifica se parametros Strech, Padding, Filters, KernelSize não diferentes. 
		// Se qualquer um dos parametros mencionados acima, na proxima convolução for diferente, então não é usado Stretch.
		// Neste caso, recompoem o Tensor com dados de Stretch, limpa Stretch, depois decompoem segundo parametros da proxima convolução.

		bool                IsStreammed;
		TensorFloat*        Stream;

		int                 Padding;
		int                 Stride;
		int                 KernelSize;
		int                 KernelDilation;
		int                 Groups;
             
		bool                IsDevice;
		uint                TotalSize;
		int                 Dimensions;
		int*                Sizes;
		float*              Grad;
		float*              Data;
	};


	struct _TensorFloatArray
	{
		int MaxCount;
		int Count;
		TensorFloat** Data;
	};


#ifdef __cplusplus
}
#endif

#endif /* TEHALTYPE */