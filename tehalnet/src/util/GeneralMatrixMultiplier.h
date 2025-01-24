 #ifndef GENERAL_MATRIX_MULTIPLIER_H
#define GENERAL_MATRIX_MULTIPLIER_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "../../include/tehalnet.h"



	void general_matrix_multiplier(bool transposed_a, bool transposed_b, int m, int n, int k, float multiplier_ab, float* a, int displacement_a, float* b, int displacement_b, float multiplier_c, float* c, int displacement_c);



#ifdef __cplusplus
}
#endif

#endif /* GENERAL_MATRIX_MULTIPLIER */