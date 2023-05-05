
#ifndef _ML_H_
#define _ML_H_

#include <stddef.h>
#include <assert.h>
#include <math.h>

#ifndef ML_MALLOC
#include <stdlib.h>
#define ML_MALLOC malloc
#endif

#ifndef ML_FREE
#define ML_FREE free
#endif

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *data;
} Matrix;

#define MATRIX_AT(m, i, j) (m).data[(i) * (m).stride + (j)]
#define MATRIX_PRINT(m) print_matrix(m, #m)

float sigmoidf(float x);
float rand_float(void);

Matrix create_matrix(size_t rows, size_t cols);
void free_matrix(Matrix m);

void randomize_matrix(Matrix m, float low, float high);
void fill_matrix(Matrix m, float value);
void apply_sigmoid(Matrix m);

void matrix_dot(Matrix dst, Matrix a, Matrix b);
void matrix_sum(Matrix dst, Matrix a);
void print_matrix(Matrix m, const char *name);
Matrix matrix_row(Matrix dst, size_t row);
void matrix_copy(Matrix dst, Matrix src);
// void matrix_sub_matrix(Matrix dst, Matrix src);

#endif // _ML_H_
