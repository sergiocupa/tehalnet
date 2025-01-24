#include "layer.h"
#include "../../../../libs/stringlib/include/stringlib.h"
#include <stdlib.h>


LayerType layer_string_to_type(String name)
{
	char* up1 = string_to_lower_copy_achar(name.Data);

	if (string_equals_char(up1, "shortcut"))
	{
		free(up1);
		return LAYER_SHORTCUT;
	}
	else if (string_equals_char(up1, "scale_channels"))
	{
		free(up1);
		return LAYER_SCALE_CHANNELS;
	}
	else if (string_equals_char(up1, "sam"))
	{
		free(up1);
		return LAYER_SAM;
	}
	else if (string_equals_char(up1, "crop"))
	{
		free(up1);
		return LAYER_CROP;
	}
	else if (string_equals_char(up1, "cost"))
	{
		free(up1);
		return LAYER_COST;
	}
	else if (string_equals_char(up1, "detection"))
	{
		free(up1);
		return LAYER_DETECTION;
	}
	else if (string_equals_char(up1, "region"))
	{
		free(up1);
		return LAYER_REGION;
	}
	else if (string_equals_char(up1, "yolo"))
	{
		free(up1);
		return LAYER_YOLO;
	}
	else if (string_equals_char(up1, "gaussian_yolo"))
	{
		free(up1);
		return LAYER_GAUSSIAN_YOLO;
	}
	else if (string_equals_char(up1, "local"))
	{
		free(up1);
		return LAYER_LOCAL;
	}
	else if (string_equals_char(up1, "conv") || string_equals_char(up1, "convolutional"))
	{
		free(up1);
		return LAYER_CONVOLUTIONAL;
	}
	else if (string_equals_char(up1, "activation"))
	{
		free(up1);
		return LAYER_ACTIVE;
	}
	else if (string_equals_char(up1, "net") || string_equals_char(up1, "network"))
	{
		free(up1);
		return LAYER_NETWORK;
	}
	else if (string_equals_char(up1, "crnn"))
	{
		free(up1);
		return LAYER_CRNN;
	}
	else if (string_equals_char(up1, "gru"))
	{
		free(up1);
		return LAYER_GRU;
	}
	else if (string_equals_char(up1, "lstm"))
	{
		free(up1);
		return LAYER_LSTM;
	}
	else if (string_equals_char(up1, "conv_lstm"))
	{
		free(up1);
		return LAYER_CONV_LSTM;
	}
	else if (string_equals_char(up1, "history"))
	{
		free(up1);
		return LAYER_HISTORY;
	}
	else if (string_equals_char(up1, "rnn"))
	{
		free(up1);
		return LAYER_RNN;
	}
	else if (string_equals_char(up1, "conn") || string_equals_char(up1, "connected"))
	{
		free(up1);
		return LAYER_CONNECTED;
	}
	else if (string_equals_char(up1, "max") || string_equals_char(up1, "maxpool"))
	{
		free(up1);
		return LAYER_MAXPOOL;
	}
	else if (string_equals_char(up1, "local_avg") || string_equals_char(up1, "local_avgpool"))
	{
		free(up1);
		return LAYER_LOCAL_AVGPOOL;
	}
	else if (string_equals_char(up1, "reorg3d"))
	{
		free(up1);
		return LAYER_REORG;
	}
	else if (string_equals_char(up1, "reorg"))
	{
		free(up1);
		return LAYER_REORG_OLD;
	}
	else if (string_equals_char(up1, "avg") || string_equals_char(up1, "avgpool"))
	{
		free(up1);
		return LAYER_AVGPOOL;
	}
	else if (string_equals_char(up1, "dropout"))
	{
		free(up1);
		return LAYER_DROPOUT;
	}
	else if (string_equals_char(up1, "lrn") || string_equals_char(up1, "normalization"))
	{
		free(up1);
		return LAYER_NORMALIZATION;
	}
	else if (string_equals_char(up1, "batchnorm"))
	{
		free(up1);
		return LAYER_BATCHNORM;
	}
	else if (string_equals_char(up1, "soft") || string_equals_char(up1, "softmax"))
	{
		free(up1);
		return LAYER_SOFTMAX;
	}
	else if (string_equals_char(up1, "contrastive"))
	{
		free(up1);
		return LAYER_CONTRASTIVE;
	}
	else if (string_equals_char(up1, "route"))
	{
		free(up1);
		return LAYER_ROUTE;
	}
	else if (string_equals_char(up1, "upsample"))
	{
		free(up1);
		return LAYER_UPSAMPLE;
	}
	else if (string_equals_char(up1, "empty") || string_equals_char(up1, "silence"))
	{
		free(up1);
		return LAYER_EMPTY;
	}
	else if (string_equals_char(up1, "implicit"))
	{
		free(up1);
		return LAYER_IMPLICIT;
	}
	else
	{
		free(up1);
		return LAYER_NONE;
	}
}

