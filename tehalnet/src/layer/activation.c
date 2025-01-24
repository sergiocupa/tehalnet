#include "activation.h"
#include "../../../../libs/stringlib/include/stringlib.h"
#include <stdlib.h> 
#include <float.h>


ActivationOption activation_string_to_type(char* name)
{
    char* up1 = string_to_lower_copy_achar(name);

    if (string_equals_char(up1, "logistic"))
    {
        free(up1);
        return ACTIVATION_LOGISTIC;
    }
    else if (string_equals_char(up1, "swish"))
    {
        free(up1);
        return ACTIVATION_SWISH;
    }
    else if (string_equals_char(up1, "mish"))
    {
        free(up1);
        return ACTIVATION_MISH;
    }
    else if (string_equals_char(up1, "hard_mish"))
    {
        free(up1);
        return ACTIVATION_HARD_MISH;
    }
    else if (string_equals_char(up1, "normalize_channels"))
    {
        free(up1);
        return ACTIVATION_NORM_CHAN;
    }
    else if (string_equals_char(up1, "normalize_channels_softmax"))
    {
        free(up1);
        return ACTIVATION_NORM_CHAN_SOFTMAX;
    }
    else if (string_equals_char(up1, "normalize_channels_softmax_maxval"))
    {
        free(up1);
        return ACTIVATION_NORM_CHAN_SOFTMAX_MAXVAL;
    }
    else if (string_equals_char(up1, "loggy"))
    {
        free(up1);
        return ACTIVATION_LOGGY;
    }
    else if (string_equals_char(up1, "relu"))
    {
        free(up1);
        return ACTIVATION_RELU;
    }
    else if (string_equals_char(up1, "relu6"))
    {
        free(up1);
        return ACTIVATION_RELU6;
    }
    else if (string_equals_char(up1, "elu"))
    {
        free(up1);
        return ACTIVATION_ELU;
    }
    else if (string_equals_char(up1, "selu"))
    {
        free(up1);
        return ACTIVATION_SELU;
    }
    else if (string_equals_char(up1, "gelu"))
    {
        free(up1);
        return ACTIVATION_GELU;
    }
    else if (string_equals_char(up1, "relie"))
    {
        free(up1);
        return ACTIVATION_RELIE;
    }
    else if (string_equals_char(up1, "plse"))
    {
        free(up1);
        return ACTIVATION_PLSE;
    }
    else if (string_equals_char(up1, "hardtan"))
    {
        free(up1);
        return ACTIVATION_HARDTAN;
    }
    else if (string_equals_char(up1, "lhtan"))
    {
        free(up1);
        return ACTIVATION_LHTAN;
    }
    else if (string_equals_char(up1, "linear"))
    {
        free(up1);
        return ACTIVATION_LINEAR;
    }
    else if (string_equals_char(up1, "ramp"))
    {
        free(up1);
        return ACTIVATION_RAMP;
    }
    else if (string_equals_char(up1, "revleaky"))
    {
        free(up1);
        return ACTIVATION_REVLEAKY;
    }
    else if (string_equals_char(up1, "leaky"))
    {
        free(up1);
        return ACTIVATION_LEAKY;
    }
    else if (string_equals_char(up1, "tanh"))
    {
        free(up1);
        return ACTIVATION_TANH;
    }
    else if (string_equals_char(up1, "stair"))
    {
        free(up1);
        return ACTIVATION_STAIR;
    }
    else
    {
        return ACTIVATION_RELU;
    }
}



// Criar seletor de ativacao e passar instancia do ativador

float activate(float x, ActivationOption a)
{
 /*   switch (a) {
    case ACTIVATION_LINEAR:
        return linear_activate(x);
    case ACTIVATION_LOGISTIC:
        return logistic_activate(x);
    case ACTIVATION_LOGGY:
        return loggy_activate(x);
    case ACTIVATION_RELU:
        return relu_activate(x);
    case ACTIVATION_ELU:
        return elu_activate(x);
    case ACTIVATION_SELU:
        return selu_activate(x);
    case ACTIVATION_GELU:
        return gelu_activate(x);
    case ACTIVATION_RELIE:
        return relie_activate(x);
    case ACTIVATION_RAMP:
        return ramp_activate(x);
    case ACTIVATION_REVLEAKY:
    case ACTIVATION_LEAKY:
        return leaky_activate(x);
    case ACTIVATION_TANH:
        return tanh_activate(x);
    case ACTIVATION_PLSE:
        return plse_activate(x);
    case ACTIVATION_STAIR:
        return stair_activate(x);
    case ACTIVATION_HARDTAN:
        return hardtan_activate(x);
    case ACTIVATION_LHTAN:
        return lhtan_activate(x);
    }*/
    return 0;
}

void activate_array_cpu_custom(float* x, const int n, const ActivationOption a)
{
    int i;
    if (a == ACTIVATION_LINEAR)
    {
    }
    else if (a == ACTIVATION_LEAKY)
    {
        for (i = 0; i < n; ++i) {
            x[i] = (x[i] > 0) ? x[i] : .1 * x[i];
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}
 

void activate_tensor(TensorFloat *input, TensorFloat *output, const ActivationOption a)
{
    int i;
    if (a == ACTIVATION_LINEAR)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_LINEAR(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_LEAKY)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_LEAKY(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_LOGISTIC)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_LOGISTIC(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_LOGGY)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_LOGGY(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_RELU)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_RELU(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_ELU)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_ELU(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_SELU)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_SELU(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_GELU)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_GELU(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_RELIE)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_RELIE(input->Data[i], output->Data[i]);
        }
    } 
    else if (a == ACTIVATION_RAMP)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_RAMP(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_LEAKY || a == ACTIVATION_REVLEAKY)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_LEAKY(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_TANH)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_TANH(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_PLSE)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_PLSE(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_STAIR)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_STAIR(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_HARDTAN)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_HARDTAN(input->Data[i], output->Data[i]);
        }
    }
    else if (a == ACTIVATION_LHTAN)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            ACTIVATE_LHTAN(input->Data[i], output->Data[i]);
        }
    }
}




void gradient_tensor(TensorFloat* input, const ActivationOption a, TensorFloat* delta)
{
    /*int i;
    if (a == ACTIVATION_LINEAR)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_LINEAR(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_LEAKY)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_LEAKY(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_LOGISTIC)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_LOGISTIC(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_LOGGY)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_LOGGY(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_RELU)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_RELU(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_ELU)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_ELU(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_SELU)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_SELU(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_GELU)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_GELU(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_RELIE)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_RELIE(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_RAMP)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_RAMP(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_LEAKY || a == ACTIVATION_REVLEAKY)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_LEAKY(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_TANH)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_TANH(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_PLSE)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_PLSE(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_STAIR)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_STAIR(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_HARDTAN)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_HARDTAN(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }
    else if (a == ACTIVATION_LHTAN)
    {
        #pragma omp parallel for
        for (i = 0; i < input->TotalSize; ++i)
        {
            float a = 0;
            GRADIENT_LHTAN(input->Data[i], a);
            delta->Data[i] *= a;
        }
    }*/
}



void activate_array(float* x, const int n, const ActivationOption a)
{
    /*int i;
    if (a == ACTIVATION_LINEAR)
    {

    }
    else if (a == ACTIVATION_LEAKY)
    {
        #pragma omp parallel for
        for (i = 0; i < n; ++i) 
        {
            x[i] = leaky_activate(x[i]);
        }
    }
    else if (a == ACTIVATION_LOGISTIC)
    {
        #pragma omp parallel for
        for (i = 0; i < n; ++i) 
        {
            x[i] = logistic_activate(x[i]);
        }
    }
    else {
        for (i = 0; i < n; ++i)
        {
            x[i] = activate(x[i], a);
        }
    }*/
}



void activate_array_swish(float* x, const int n, float* output_sigmoid, float* output)
{
    /*int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) 
    {
        float x_val = x[i];
        float sigmoid = logistic_activate(x_val);
        output_sigmoid[i] = sigmoid;
        output[i] = x_val * sigmoid;
    }*/
}

// https://github.com/digantamisra98/Mish
void activate_array_mish(float* x, const int n, float* activation_input, float* output)
{
    //const float MISH_THRESHOLD = 20;
    //int i;
    //#pragma omp parallel for
    //for (i = 0; i < n; ++i) 
    //{
    //    float x_val = x[i];
    //    activation_input[i] = x_val;    // store value before activation
    //    output[i] = x_val * tanh_activate(softplus_activate(x_val, MISH_THRESHOLD));
    //}
}

static float hard_mish_yashas(float x)
{
    if (x > 0)
        return x;
    if (x > -2)
        return x * x / 2 + x;
    return 0;
}

void activate_array_hard_mish(float* x, const int n, float* activation_input, float* output)
{
    int i;

    #pragma omp parallel for
    for (i = 0; i < n; ++i)
    {
        float x_val = x[i];
        activation_input[i] = x_val;    // store value before activation
        //output[i] = hard_mish_yashas(x_val);
    }
}

void activate_array_normalize_channels(float* x, const int n, int batch, int channels, int wh_step, float* output)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) 
    {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        const float eps = 0.0001;
        if (i < size) 
        {
            float sum = eps;
            int k;
            for (k = 0; k < channels; ++k) 
            {
                float val = x[wh_i + k * wh_step + b * wh_step * channels];
                if (val > 0) sum += val;
            }
            for (k = 0; k < channels; ++k) 
            {
                float val = x[wh_i + k * wh_step + b * wh_step * channels];
                if (val > 0) val = val / sum;
                else val = 0;
                output[wh_i + k * wh_step + b * wh_step * channels] = val;
            }
        }
    }
}

void activate_array_normalize_channels_softmax(float* x, const int n, int batch, int channels, int wh_step, float* output, int use_max_val)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) 
    {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        const float eps = 0.0001;
        if (i < size)
        {
            float sum = eps;
            float max_val = -FLT_MAX;
            int k;
            if (use_max_val) 
            {
                for (k = 0; k < channels; ++k) {
                    float val = x[wh_i + k * wh_step + b * wh_step * channels];
                    if (val > max_val || k == 0) max_val = val;
                }
            }
            else
                max_val = 0;

            for (k = 0; k < channels; ++k) 
            {
                float val = x[wh_i + k * wh_step + b * wh_step * channels];
                sum += expf(val - max_val);
            }
            for (k = 0; k < channels; ++k) 
            {
                float val = x[wh_i + k * wh_step + b * wh_step * channels];
                val = expf(val - max_val) / sum;
                output[wh_i + k * wh_step + b * wh_step * channels] = val;
            }
        }
    }
}

void gradient_array_normalize_channels_softmax(float* x, const int n, int batch, int channels, int wh_step, float* delta)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) 
    {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        if (i < size) 
        {
            float grad = 0;
            int k;

            for (k = 0; k < channels; ++k) 
            {
                const int index = wh_i + k * wh_step + b * wh_step * channels;
                float out = x[index];
                float d = delta[index];
                grad += out * d;
            }

            for (k = 0; k < channels; ++k) 
            {
                const int index = wh_i + k * wh_step + b * wh_step * channels;
                float d = delta[index];
                d = d * grad;
                delta[index] = d;
            }
        }
    }
}

void gradient_array_normalize_channels(float* x, const int n, int batch, int channels, int wh_step, float* delta)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i)
    {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        if (i < size) 
        {
            float grad = 0;
            int k;
            for (k = 0; k < channels; ++k)
            {
                const int index = wh_i + k * wh_step + b * wh_step * channels;
                float out = x[index];
                float d = delta[index];
                grad += out * d;
            }

            for (k = 0; k < channels; ++k) 
            {
                const int index = wh_i + k * wh_step + b * wh_step * channels;
                if (x[index] > 0) {
                    float d = delta[index];
                    d = d * grad;
                    delta[index] = d;
                }
            }
        }
    }
}

// TO-DO: implementar codigo comentado...
void gradient_array_hard_mish(const int n, const float* activation_input, float* delta)
{
    int i;

    #pragma omp parallel for
    for (i = 0; i < n; ++i) 
    {
        float inp = activation_input[i];
        //delta[i] *= hard_mish_yashas_grad(inp);
    }
}

void gradient_array_mish(const int n, const float* activation_input, float* delta)
{
    int i;

    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        const float MISH_THRESHOLD = 20.0f;

        // implementation from TensorFlow: https://github.com/tensorflow/addons/commit/093cdfa85d334cbe19a37624c33198f3140109ed
        // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
        /*float inp = activation_input[i];
        const float sp = softplus_activate(inp, MISH_THRESHOLD);
        const float grad_sp = 1 - exp(-sp);
        const float tsp = tanh(sp);
        const float grad_tsp = (1 - tsp * tsp) * grad_sp;
        const float grad = inp * grad_tsp + tsp;
        delta[i] *= grad;*/


        //float x = activation_input[i];
        //float d = 2 * expf(x) + expf(2 * x) + 2;
        //float w = 4 * (x + 1) + 4 * expf(2 * x) + expf(3 * x) + expf(x)*(4 * x + 6);
        //float derivative = expf(x) * w / (d * d);
        //delta[i] *= derivative;
    }
}


// https://github.com/BVLC/caffe/blob/04ab089db018a292ae48d51732dd6c66766b36b6/src/caffe/layers/swish_layer.cpp#L54-L56
void gradient_array_swish(const float* x, const int n, const float* sigmoid, float* delta)
{
    int i;

    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float swish = x[i];
        delta[i] *= swish + sigmoid[i] * (1 - swish);
    }
}


// TO-DO: implementar...
float gradient(float x, ActivationOption a)
{
    //switch (a) {
    //case ACTIVATION_LINEAR:
    //    return linear_gradient(x);
    //case ACTIVATION_LOGISTIC:
    //    return logistic_gradient(x);
    //case ACTIVATION_LOGGY:
    //    return loggy_gradient(x);
    //case ACTIVATION_RELU:
    //    return relu_gradient(x);
    //case ACTIVATION_RELU6:
    //    return relu6_gradient(x);
    //case ACTIVATION_NORM_CHAN:
    //    //return relu_gradient(x);
    //case ACTIVATION_NORM_CHAN_SOFTMAX_MAXVAL:
    //    //...
    //case ACTIVATION_NORM_CHAN_SOFTMAX:
    //    error("Error: should be used custom NORM_CHAN or NORM_CHAN_SOFTMAX-function for gradient", DARKNET_LOC);
    //case ACTIVATION_ELU:
    //    return elu_gradient(x);
    //case ACTIVATION_SELU:
    //    return selu_gradient(x);
    //case ACTIVATION_GELU:
    //    return gelu_gradient(x);
    //case ACTIVATION_RELIE:
    //    return relie_gradient(x);
    //case ACTIVATION_RAMP:
    //    return ramp_gradient(x);
    //case ACTIVATION_REVLEAKY:
    //case ACTIVATION_LEAKY:
    //    return leaky_gradient(x);
    //case ACTIVATION_TANH:
    //    return tanh_gradient(x);
    //case ACTIVATION_PLSE:
    //    return plse_gradient(x);
    //case ACTIVATION_STAIR:
    //    return stair_gradient(x);
    //case ACTIVATION_HARDTAN:
    //    return hardtan_gradient(x);
    //case ACTIVATION_LHTAN:
    //    return lhtan_gradient(x);
    //}
    return 0;
}


