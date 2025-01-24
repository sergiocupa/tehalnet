#include "local_layer.h"
#include "activation.h"
#include "../util/GeneralMatrixMultiplier.h"
#include <string.h>


//int local_out_height(LayerInfo *i)
//{
//	int h = i->Image.Height;
//	if (!i->Padding) h -= i->FilterSize;
//	else h -= 1;
//	return h / i->Stride + 1;
//}
//
//int local_out_width(LayerInfo* i)
//{
//	int w = i->Image.Width;
//	if (!i->Padding) w -= i->FilterSize;
//	else w -= 1;
//	return w / i->Stride + 1;
//}


void forward_local_layer(LocalLayer* l, NetworkState* state)
{
	//int out_h   = local_out_height(l);
	//int out_w   = local_out_width(l);
	//int w       = l->Info.Image.Width;
	//int h       = l->Info.Image.Height;
	//int ch      = l->Info.Image.Channels;
	//int w_h_ch  = w * h * ch;
	//int s2      = l->Info.FilterSize * l->Info.FilterSize;
	//int s2_ch   = s2 * l->Info.Image.Channels;
	//int s2_ch_f = s2_ch * l->Info.Filters;
	//int i, j;
	//int locations = out_h * out_w;

	//for (i = 0; i < l->Info.Batch; ++i)
	//{
	//	// TO-DO: pegar do arquivo blas.c
	//	//copy_cpu(l.outputs, l.biases, 1, l.output + i * l.outputs, 1);
	//}

	//for (i = 0; i < l->Info.Batch; ++i)
	//{
	//	float* input = state->Input.Data + i * w_h_ch;

	//	// TO-DO: buscar e implementar
	//	//im2col_cpu(input, ch, h, w, s, l->Info.Stride, l->Info.Padding, l->ColImage->Data);

	//	float* output = l->Output.Data + i * l->Output.TotalSize;

	//	for (j = 0; j < locations; ++j) 
	//	{
	//		float* a = l->Weights->Data + j * s2_ch_f;
	//		float* b = l->ColImage->Data + j;
	//		float* c = output + j;

	//		int m = l->Info.Filters;
	//		int n = 1;
	//		int k = s2_ch;

	//		general_matrix_multiplier(false,false,m,n,k,1,a,k,b,locations,1,c,locations);
	//	}
	//}
	//activate_array(l->Output.Data, l->Output.TotalSize * l->Info.Batch, l->Info.Activation);
}


void backward_local_layer(LocalLayer* l, NetworkState* state)
{
	//int i, j;

	//int locations = l->Output.Sizes[1] * l->Output.Sizes[2]; // Width and Height
	//int s2        = l->Info.FilterSize * l->Info.FilterSize;
	//int s2ch      = s2 * l->Info.Image.Channels;
	//int s2_ch_f   = s2ch * l->Info.Filters;
	//int w_h_ch    = locations * l->Info.Image.Channels;

	//// TO-DO: buscar e implementar
	////gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);

	//for (i = 0; i < l->Info.Batch; ++i)
	//{
	//	// TO-DO: buscar e implementar
	//	//axpy_cpu(l->Output.TotalSize, 1, l->TraiInfo.Delta.Size + i * l->Output.TotalSize, 1, l->BiasUpdates->Size, 1);
	//}

	//for (i = 0; i < l->Info.Batch; ++i)
	//{
	//	float* input = state->Input.Data + i * w_h_ch;

	//	// TO-DO: buscar e implementar
	//	//im2col_cpu(input, l.c, l.h, l.w, l.size, l.stride, l.pad, l.col_image);

	//	for (j = 0; j < locations; ++j)
	//	{
	//		float* a = l->TraiInfo.Delta.Data + i * l->Output.TotalSize + j;
	//		float* b = l->ColImage->Data + j;
	//		float* c = l->WeightUpdates->Data + j * s2_ch_f;
	//		int m = l->Info.Filters;
	//		int n = s2ch;
	//		int k = 1;

	//		general_matrix_multiplier(false, true, m, n, k, 1, a, locations, b, locations, 1, c, n);
	//	}

	//	if (state->Delta.Size > 0)
	//	{
	//		for (j = 0; j < locations; ++j)
	//		{
	//			float* a = l->Weights + j * s2_ch_f;
	//			float* b = state->Delta.Data + i * l->Output.TotalSize + j;
	//			float* c = l->ColImage->Data + j;

	//			int m = s2ch;
	//			int n = 1;
	//			int k = l->Info.Filters;

	//			general_matrix_multiplier(true, false, m, n, k, 1, a, m, b, locations, 0, c, locations);
	//		}

	//		// TO-DO: buscar e implementar
	//		//col2im_cpu(l.col_image, l.c, l.h, l.w, l.size, l.stride, l.pad, state.delta + i * l.c * l.h * l.w);
	//	}
	//}
}


void update_local_layer(LocalLayer* l, int batch, float learning_rate, float momentum, float decay)
{
	//int locations = l->Output.Sizes[1] * l->Output.Sizes[2]; // Width and Height
	//int s2        = l->Info.FilterSize * l->Info.FilterSize;
	//int s2ch      = s2 * l->Info.Image.Channels;
	//int s2chf     = s2ch * l->Info.Filters;
	//int size      = s2chf * locations;

	//// TO-DO: buscar e implementar
	////axpy_cpu(l.outputs, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
	////scal_cpu(l.outputs, momentum, l.bias_updates, 1);

	////axpy_cpu(size, -decay * batch, l.weights, 1, l.weight_updates, 1);
	////axpy_cpu(size, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
	////scal_cpu(size, momentum, l.weight_updates, 1);
}



LocalLayer* make_local_layer(LayerInfo* param)
{
	//int i;

	//LocalLayer* l = (LocalLayer*)TehLib.allocate(false, sizeof(LocalLayer));
	//memset(l, 0, sizeof(LocalLayer));

	//l->Info.Type = LAYER_CONVOLUTIONAL;
	//l->Info.Activation = param->Activation;

	//l->Info.Image.Height = param->Image.Height;
	//l->Info.Image.Width = param->Image.Width;
	//l->Info.Image.Channels = param->Image.Channels;
	//l->Info.Filters = param->Filters;
	//l->Info.FilterSize = param->FilterSize;
	//l->Info.Batch = param->Batch;
	//l->Info.Stride = param->Stride;
	//l->Info.Padding = param->Padding;

	//int  out_h = local_out_height(&l->Info);
	//int  out_w = local_out_width(&l->Info);
	//int  locat = out_h * out_w;
	//int  outz = locat * param->Filters;
	//long sz = param->Image.Channels * param->Filters * param->FilterSize * param->FilterSize * locat;

	//TehLib.init_tensor_imagem(&l->Output, TehLib.UseDevice, out_w, out_h, l->Info.Filters);
	//TehLib.init_buffer(l->Weights, TehLib.UseDevice, sz);
	//TehLib.init_buffer(l->WeightUpdates, TehLib.UseDevice, sz);
	//TehLib.init_buffer(l->Biases, TehLib.UseDevice, l->Output.TotalSize);
	//TehLib.init_buffer(l->BiasUpdates, TehLib.UseDevice, l->Output.TotalSize);


	//// float scale = 1./sqrt(size*size*c);
	//float scale = sqrt(2. / (l->Info.FilterSize * l->Info.FilterSize * l->Info.Image.Channels));
	//for (i = 0; i < l->Info.Image.Channels * l->Info.Filters * l->Info.FilterSize * l->Info.FilterSize; ++i)
	//{
	//	// TO-DO: verificar onde esta implementacao de rand_uniform()
	//	//l->Weights->Data[i] = scale * rand_uniform(-1, 1);
	//}

	//int col_size = out_h * out_w * l->Info.FilterSize * l->Info.FilterSize * l->Info.Image.Channels;
	//TehLib.init_buffer(l->ColImage, TehLib.UseDevice, col_size);
	//TehLib.init_buffer(&l->TraiInfo.Delta, TehLib.UseDevice, (l->Info.Batch * out_h * out_w * l->Info.Filters));

	//l->Forward  = forward_local_layer;
	//l->Backward = backward_local_layer;
	//l->Update   = update_local_layer;
}
