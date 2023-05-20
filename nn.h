#ifndef _NN_H_
#define _NN_H_

#include "matrix.h"

typedef struct
{
    size_t count;
    Matrix *ws; // weights
    Matrix *bs; // Biases
    Matrix *as; // activations (count + 1) (1 for the input)
} NN;

#define ARRAY_LEN(a) sizeof((a)) / sizeof((a)[0])
#define NN_PRINT(nn) nn_print(nn, #nn)
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
void nn_free(NN nn);

void nn_randomize(NN nn, int minimum, int maximum);
void nn_zero(NN nn);

void nn_forward(NN nn);
float nn_cost(NN nn, Matrix ti, Matrix to);
void nn_finite_diff(NN nn, NN g, float eps, Matrix ti, Matrix to);
void nn_backprop(NN nn, NN g, Matrix ti, Matrix to);
void nn_learn(NN n, NN g, float rate);

#endif // _NN_H_
