#ifndef ACTIVATION_H
#define ACTIVATION_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "../../include/tehalnet.h"

    #define _USE_MATH_DEFINES
    #include <math.h>

    // ACTIVATE_LEAKY or ACTIVATE_LEAKY_RELU
    //double leaky_relu(double x, double alpha) {
    //    return (x > 0) ? x : alpha * x;
    //}

    #define ACTIVATE_LINEAR(in, out)              out  = in
    #define ACTIVATE_LEAKY(in, out)               out  = (in > 0) ? in : .1f * in
    #define ACTIVATE_LOGISTIC(in, out)            out  = 1.f / (1.f + expf(-in))
    #define ACTIVATE_LOGGY(in, out)               out  = 2.f / (1.f + expf(-in)) - 1
    #define ACTIVATE_RELU(in, out)                out  = in * (in > 0)
    #define ACTIVATE_ELU(in, out)                 out  = (in >= 0) * in + (in < 0) * (expf(in) - 1)
    #define ACTIVATE_SELU(in, out)                out  = (in >= 0) * 1.0507f * in + (in < 0) * 1.0507f * 1.6732f * (expf(in) - 1)
    #define ACTIVATE_GELU(in, out)                out  = (0.5 * in * (1 + tanhf(0.797885 * in + 0.035677 * powf(in, 3))))
    #define ACTIVATE_RELIE(in, out)               out  = (in > 0) ? in : .01f * in
    #define ACTIVATE_RAMP(in, out)                out  = in * (in > 0) + .1f * in
    #define ACTIVATE_TANH(in, out)                out  = (2 / (1 + expf(-2 * in)) - 1)
    #define ACTIVATE_PLSE(in, out)                out  = (in < -4) ? (.01f * (in + 4)) : ((in > 4) ? (.01f * (in - 4) + 1) : (.125f * in + .5f))
    #define ACTIVATE_HARDTAN(in, out)             out  = (in < -1) ? -1 : ((in > 1) ? 1 : in)
    #define ACTIVATE_LHTAN(in, out)               out  = (in < 0) ? (.001f * in) : ((in > 1) ? (.001f * (in - 1) + 1) : in)
    #define ACTIVATE_SOFTPLUS(in, out, threshold) out = (in > threshold) ? in : ((in < -threshold) ? (expf(in)) : (logf(expf(in) + 1)))
    #define ACTIVATE_STAIR(in, out) int calc = floorf(in); out = (calc % 2 == 0) ? floorf(in / 2.f) : ((in - calc) + floorf(in / 2.f))

    #define GRADIENT_LINEAR(in, out)   out = 1
    #define GRADIENT_LHTAN(in, out)    out = (in > 0 && in < 1) ? 1 : .001f
    #define GRADIENT_HARDTAN(in, out)  out = (in > -1 && in < 1) ? 1 : 0
    #define GRADIENT_LOGISTIC(in, out) out = (1 - in) * in
    #define GRADIENT_LOGGY(in, out)    float y = (in + 1.f) / 2.f; out = 2 * (1 - y) * y
    #define GRADIENT_STAIR(in, out)    out = (floor(in) == in) ? 0 : 1.0f
    #define GRADIENT_RELU(in, out)     out = (in > 0)
    #define GRADIENT_RELU6(in, out)    out = (in > 0 && in < 6)
    #define GRADIENT_ELU(in, out)      out = (in >= 0) + (in < 0) * (in + 1)
    #define GRADIENT_SELU(in, out)     out = (in >= 0) * 1.0507f + (in < 0) * (in + 1.0507f * 1.6732f)
    #define GRADIENT_RELIE(in, out)    out = (in > 0) ? 1 : .01f
    #define GRADIENT_RAMP(in, out)     out = (in > 0) + .1f
    #define GRADIENT_LEAKY(in, out)    out = (in > 0) ? 1 : .1f
    #define GRADIENT_TANH(in, out)     out =  1 - in * in
    #define GRADIENT_SECH(in, out)     out = 2 / (expf(in) + expf(-in))
    #define GRADIENT_GELU(in, out)     const float x3 = powf(in, 3); out = 0.5 * tanhf(0.0356774 * x3 + 0.797885 * in) + (0.0535161 * x3 + 0.398942 * in) * powf(sech(0.0356774 * x3 + 0.797885 * in), 2) + 0.5
    #define GRADIENT_PLSE(in, out)     out = (in < 0 || in > 1) ? .01f : .125f


   // static inline float lhtan_gradient(float x)
   // {
   //     if (x > 0 && x < 1) return 1;
   //     return .001f;
   // }

   // static inline float hardtan_gradient(float x)
   // {
   //     if (x > -1 && x < 1) return 1;
   //     return 0;
   // }
   // static inline float linear_gradient(float x) { return 1; }
   // static inline float logistic_gradient(float x) { return (1 - x) * x; }
   // static inline float loggy_gradient(float x)
   // {
   //     float y = (x + 1.f) / 2.f;
   //     return 2 * (1 - y) * y;
   // }
   // static inline float stair_gradient(float x)
   // {
   //     if (floor(x) == x) return 0;
   //     return 1.0f;
   // }
   // static inline float relu_gradient(float x) { return (x > 0); }
   // static inline float relu6_gradient(float x) { return (x > 0 && x < 6); }
   // static inline float elu_gradient(float x) { return (x >= 0) + (x < 0) * (x + 1); }
   // static inline float selu_gradient(float x) { return (x >= 0) * 1.0507f + (x < 0) * (x + 1.0507f * 1.6732f); }
   // static inline float relie_gradient(float x) { return (x > 0) ? 1 : .01f; }
   // static inline float ramp_gradient(float x) { return (x > 0) + .1f; }
   // static inline float leaky_gradient(float x) { return (x > 0) ? 1 : .1f; }
   // static inline float tanh_gradient(float x) { return 1 - x * x; }

   // static inline float sech(float x) { return 2 / (expf(x) + expf(-x)); }
   // static inline float gelu_gradient(float x) 
   // {
   //     const float x3 = powf(x, 3);
   //     return 0.5 * tanhf(0.0356774 * x3 + 0.797885 * x) + (0.0535161 * x3 + 0.398942 * x) * powf(sech(0.0356774 * x3 + 0.797885 * x), 2) + 0.5;
   // }
   // static inline float plse_gradient(float x) { return (x < 0 || x > 1) ? .01f : .125f; }




   // static inline float linear_activate(float x) { return x; }
   // static inline float logistic_activate(float x) { return 1.f / (1.f + expf(-x)); }
   // static inline float loggy_activate(float x) { return 2.f / (1.f + expf(-x)) - 1; }
   // static inline float relu_activate(float x) { return x * (x > 0); }
   //// static inline float relu6_activate(float x) { return min_val_cmp(max_val_cmp(x, 0), 6); }
   // static inline float elu_activate(float x) { return (x >= 0) * x + (x < 0) * (expf(x) - 1); }
   // static inline float selu_activate(float x) { return (x >= 0) * 1.0507f * x + (x < 0) * 1.0507f * 1.6732f * (expf(x) - 1); }
   // static inline float relie_activate(float x) { return (x > 0) ? x : .01f * x; }
   // static inline float ramp_activate(float x) { return x * (x > 0) + .1f * x; }
   // static inline float leaky_activate(float x) { return (x > 0) ? x : .1f * x; }
   // //static inline float tanh_activate(float x){return (expf(2*x)-1)/(expf(2*x)+1);}
   // static inline float tanh_activate(float x) { return (2 / (1 + expf(-2 * x)) - 1); }
   // static inline float gelu_activate(float x) { return (0.5 * x * (1 + tanhf(0.797885 * x + 0.035677 * powf(x, 3)))); }



   // static inline float softplus_activate(float x, float threshold)
   // {
   //     if (x > threshold) return x;                // too large
   //     else if (x < -threshold) return expf(x);    // too small
   //     return logf(expf(x) + 1);
   // }

   // static inline float plse_activate(float x)
   // {
   //     if (x < -4) return .01f * (x + 4);
   //     if (x > 4)  return .01f * (x - 4) + 1;
   //     return .125f * x + .5f;
   // }


   // static inline float lhtan_activate(float x)
   // {
   //     if (x < 0) return .001f * x;
   //     if (x > 1) return .001f * (x - 1) + 1;
   //     return x;
   // }

   // static inline float stair_gradient(float x)
   // {
   //     if (floor(x) == x) return 0;
   //     return 1.0f;
   // }

  

   // static inline float stair_activate(float x)
   // {
   //     int n = floorf(x);
   //     if (n % 2 == 0) return floorf(x / 2.f);
   //     else return (x - n) + floorf(x / 2.f);
   // }

   // static inline float hardtan_activate(float x)
   // {
   //     if (x < -1) return -1;
   //     if (x > 1) return 1;
   //     return x;
   // }


	ActivationOption activation_string_to_type(char* value);


    void gradient_tensor(TensorFloat* input, const ActivationOption a, TensorFloat* delta);
    void activate_tensor(TensorFloat* input, TensorFloat* output, const ActivationOption a);

    void activate_array_cpu_custom(float* x, const int n, const ActivationOption a);
    void activate_array(float* x, const int n, const ActivationOption a);
    void activate_array_swish(float* x, const int n, float* output_sigmoid, float* output);
    void activate_array_mish(float* x, const int n, float* activation_input, float* output);
    static float hard_mish_yashas(float x);
    void activate_array_hard_mish(float* x, const int n, float* activation_input, float* output);
    void activate_array_normalize_channels(float* x, const int n, int batch, int channels, int wh_step, float* output);
    void activate_array_normalize_channels_softmax(float* x, const int n, int batch, int channels, int wh_step, float* output, int use_max_val);
    void gradient_array_normalize_channels_softmax(float* x, const int n, int batch, int channels, int wh_step, float* delta);
    void gradient_array_normalize_channels(float* x, const int n, int batch, int channels, int wh_step, float* delta);
    void gradient_array_hard_mish(const int n, const float* activation_input, float* delta);
    void gradient_array_mish(const int n, const float* activation_input, float* delta);
    void gradient_array_swish(const float* x, const int n, const float* sigmoid, float* delta);
    void gradient_array(const float* x, const int n, const ActivationOption a, float* delta);

#ifdef __cplusplus
}
#endif

#endif /* ACTIVATION */