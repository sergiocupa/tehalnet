#ifndef TEHALNET_H
#define TEHALNET_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "platform.h"
    #include "../../tehaltype/include/tehaltype.h"

	typedef enum   _OptimizerOption      OptimizerOption;
	typedef enum   _LearninRatePolicy    LearninRatePolicy;
	typedef struct _Box                  Box;
	typedef struct _OptimizerInfo        OptimizerInfo;
	typedef struct _DataAugmentationInfo DataAugmentationInfo;
	typedef struct _TrainInfo            TrainInfo;
	typedef struct _BatchNormalize       BatchNormalizeInfo;
	typedef struct _NetworkInfo          NetworkInfo;
	typedef struct _NetworkState         NetworkState;
	typedef struct _LayerInfo            LayerInfo;
	typedef struct _Layer                Layer;
	typedef struct _TehalLib             TehalLib;

	typedef enum _LayerType
	{
		LAYER_NONE            = 0,
		LAYER_CONVOLUTIONAL   = 1,
		LAYER_DECONVOLUTIONAL = 2,
		LAYER_CONNECTED       = 3,
		LAYER_MAXPOOL         = 4,
		LAYER_LOCAL_AVGPOOL   = 5,
		LAYER_SOFTMAX         = 6,
		LAYER_DETECTION       = 7,
		LAYER_DROPOUT         = 8,
		LAYER_CROP            = 9,
		LAYER_ROUTE           = 10,
		LAYER_COST            = 11,
		LAYER_NORMALIZATION   = 12,
		LAYER_AVGPOOL         = 13,
		LAYER_LOCAL           = 14,
		LAYER_SHORTCUT        = 15,
		LAYER_SCALE_CHANNELS  = 16,
		LAYER_SAM             = 17,
		LAYER_ACTIVE          = 18,
		LAYER_RNN             = 19,
		LAYER_GRU             = 20,
		LAYER_LSTM            = 21,
		LAYER_CONV_LSTM       = 22,
		LAYER_HISTORY         = 23,
		LAYER_CRNN            = 24,
		LAYER_BATCHNORM       = 25,
		LAYER_NETWORK         = 26,
		LAYER_XNOR            = 27,
		LAYER_REGION          = 28,
		LAYER_YOLO            = 29,
		LAYER_GAUSSIAN_YOLO   = 30,
		LAYER_ISEG            = 31,
		LAYER_REORG           = 32,
		LAYER_REORG_OLD       = 33,
		LAYER_UPSAMPLE        = 34,
		LAYER_LOGXENT         = 35,
		LAYER_L2NORM          = 36,
		LAYER_EMPTY           = 37,
		LAYER_CONTRASTIVE     = 38,
		LAYER_IMPLICIT        = 39
	}
	LayerType;

	typedef enum _ActivationOption
	{
		ACTIVATION_LINEAR                   = 0,
		ACTIVATION_LOGISTIC                 = 1, 
		ACTIVATION_RELU                     = 2, 
		ACTIVATION_RELU6                    = 3, 
		ACTIVATION_RELIE                    = 4, 
		ACTIVATION_RAMP                     = 5, 
		ACTIVATION_TANH                     = 6, 
		ACTIVATION_PLSE                     = 7, 
		ACTIVATION_REVLEAKY                 = 8, 
		ACTIVATION_LEAKY                    = 9,
		ACTIVATION_ELU                      = 10,
		ACTIVATION_LOGGY                    = 11, 
		ACTIVATION_STAIR                    = 12, 
		ACTIVATION_HARDTAN                  = 13, 
		ACTIVATION_LHTAN                    = 14, 
		ACTIVATION_SELU                     = 15, 
		ACTIVATION_GELU                     = 16, 
		ACTIVATION_SWISH                    = 17,
		ACTIVATION_MISH                     = 18, 
		ACTIVATION_HARD_MISH                = 19,
		ACTIVATION_NORM_CHAN                = 20,
		ACTIVATION_NORM_CHAN_SOFTMAX        = 21, 
		ACTIVATION_NORM_CHAN_SOFTMAX_MAXVAL = 22
	}
	ActivationOption;


	typedef enum _OptimizerOption
	{
		OPTIMIZER_NONE = 0,
		OPTIMIZER_ADAM = 1
	};


	typedef enum _LearninRatePolicy
	{
		CONSTANT = 0, 
		STEP     = 1,
		EXP      = 2,
		POLY     = 3, 
		STEPS    = 4, 
		SIG      = 5, 
		RANDOM   = 6, 
		SGDR     = 7
	};

	typedef struct _Box 
	{
		float x, y, w, h;
	};


	
	typedef struct _DataAugmentationInfo
	{
		int Angle;
		int Saturation;
		int Exposure;
		int Hue;
		int Blur;
		int MinCrop;
		int MaxCrop;
		int Aspect;
		int LetterBox;
		int Cutmix;
		int Mosaic;
		int MosaicBound;
		int Jitter;
		int Random;
		int AdversarialLr;
		int Attention;
		int GaussianNoise;
	};


	typedef struct _OptimizerInfo
	{
		OptimizerOption   Type;
		TensorFloat      *M;
		TensorFloat      *V;
		BufferFloat      *BiasM;
		BufferFloat      *BiasV;
		BufferFloat      *ScaleM;
		BufferFloat      *ScaleV;
		float             Momentum;
		float             Decay;
		float             LearningRate;
		int               BurnIn;
		int               MaxBatches;
		LearninRatePolicy Policy;
		float             Power;
		int               BatchesPerCycle;// sgdr_cycle
		int               BatchesCycleMult;// sgdr_mult
		int*              Steps;
		float*            Scales;
		float             LabelSmoothEps;
	};


	typedef struct _TrainInfo
	{
		bool                 IsEnabled;
		int                  Batch;
		float                Epoch;
		bool                 IsBatchNormalize;
		bool                 AssistedExcitation;
		bool                 Antialiasing;
		TensorFloat* Delta;
		TensorFloat* WeightUpdates;
		TensorFloat* BiasUpdates;
		TensorFloat* WeightsEma;
		TensorFloat* BiasesEma;
		OptimizerInfo        Optimizer;
		DataAugmentationInfo Augmentation;

		// Para entrada de imagens sequencias, video
		bool                 IsTrack;
		int                  TimeSteps;
		int                  AugmentSpeed;
		int                  SequentialSubdivisions;
		float                SeqScales;
	};


	typedef struct _NetworkInfo
	{
		ImageFloat ImageSize;
		TrainInfo  TraiInfo;
		uint64_t* Seen;
		int        Batch;
		int        Subdivisions;
		int        MaxBatches;
		int        NumBoxes;
		int        Truth;
	};


	typedef struct _NetworkState
	{
		BufferFloat  Truth;
		BufferFloat  Input;
		TensorFloat* Delta;
		BufferFloat  Workspace;
		bool         IsTrain;
		int          Index;
		NetworkInfo  Net;
	};


	typedef struct _LayerInfo
	{
		LayerType        Type;
		bool             Antialiasing;
		int              Truths;
		int              MaxBoxes;
		int              Batch;
		int              Stride;
		int              WaitStremID;
		int              Index;
		int              Groups;
		int              Filters;
		int              KernelSize;
		int              KernelDilation;
		int              Padding;
		bool             Deform;
		int              LearningRateScale;
		bool             IsBatchNormalize;
		OptimizerOption  Optimizer;
		ActivationOption Activation;
		TensorInfo       Input;
		TensorInfo       Output;
		int              DeviceIndex;
		bool             IsTrainMode;
		int              AssistedExcitation;
		int              Sway;
		bool             Reverse;
		int              MaxChartLoss;
	};


	typedef struct _BatchNormalize
	{
		bool         IsEnabled;
		float        Momentum;
		float        Eps;
		bool         Affine;
		bool         TrackRunningStats;
		BufferFloat *Scales;
		BufferFloat *ScalesEma;
		BufferFloat *ScaleUpdates;
		BufferFloat *Mean;
		BufferFloat *Variance;
		BufferFloat *MeanDelta;
		BufferFloat *VarianceDelta;
		BufferFloat *RunningMean;
		BufferFloat *RunningVariance;
	};

	typedef struct _Layer
	{
		int                 Index;
		int                 ParentIndex;
		TensorFloatArray    Elements;
		TrainInfo           TraiInfo;
		LayerInfo           Info;
		TensorFloat*        Input;
		TensorFloat*        Output;
		TensorFloat*        InputMemo;// Armazena endereco do INPUT, para nao perder durante substituicoes nas convoluções
		TensorFloat*        OutputMemo;// Armazena endereco do INPUT, para nao perder durante substituicoes nas convoluções
		TensorFloat*        Weights;

		bool                BuildOutputTensor;



		void(*Forward)      (struct layer, struct network_state);
		void(*Backward)     (struct layer, struct network_state);
		void(*Update)       (struct layer, int, float, float, float);
		void(*ForwardGpu)   (struct layer, struct network_state);
		void(*BackwardGpu)  (struct layer, struct network_state);
		void(*UpdateGpu)    (struct layer, int, float, float, float, float);
	};

	typedef struct _TehalLib
	{
		bool UseDevice;
		int  DeviceIndex;

		void         (*copy_int)             (int* source, int* dest, uint source_size, uint dest_size, bool source_device, bool dest_device);
		void         (*copy_float)           (float* source, float* dest, uint source_size, uint dest_size, bool source_device, bool dest_device);
		void         (*copy)                 (TensorFloat* source, TensorFloat* dest);
		void*        (*allocate)             (bool is_device, uint size);
		void         (*deallocate)           (bool is_device, void*);
		TensorFloat* (*create_tensor_imagem) (bool is_device, int width, int height, int channels);
		void         (*init_tensor_imagem)   (TensorFloat* tensor, bool is_device, int width, int height, int channels);
		TensorFloat* (*create_tensor)        (bool is_device, int dim, int* sizes);
		void         (*init_tensor)          (TensorFloat* tensor, bool is_device, int dim, int* sizes);
		void         (*release_tensor)       (TensorFloat* tensor);
		void         (*free_tensor)          (TensorFloat* tensor);
		BufferFloat* (*create_buffer)        (bool is_device, uint size, bool set_value, float value);
		void         (*init_buffer)          (BufferFloat* buffer, bool is_device, uint size, bool set_value, float value);

	};




	TehalLib TehLib;



	//void init_tensor_array(TensorFloatArray* ar);
	//TensorFloatArray* create_tensor_array();
	//void tensor_array_add(TensorFloatArray* _this, TensorFloat* item);
	//void tensor_array_add_index(TensorFloatArray* _this, TensorFloat* item, int index);


	// No nome da biblioteca, nao colocar a extensão do arquivo
	PLATFORM_API bool teh_init(bool use_device, int device_index, char* devide_lib);

	//PLATFORM_API NetworkInfo teh_load_config(const char* path_file_config, const char* path_file_weights);

	//PLATFORM_API void init_tensor_info(TensorInfo* reference, TensorInfo* info);


	//PLATFORM_API void init_default_layer(LayerInfo* layer);


	//PLATFORM_API Layer* make_convolutional_layer(TensorInfo input, int kernel_size, int kernel_dilation, int output_filters, int padding, int stride, int groups, ActivationOption activation, bool use_batch_normalize, bool is_train_mode, int steps);




#ifdef __cplusplus
}
#endif

#endif /* TEHALNET */