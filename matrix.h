#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *data;
} Matrix;

#define MATRIX_AT(m, i, j) (m).data[(i) * (m).stride + (j)]
#define MATRIX_PRINT(m) print_matrix(m, #m, 0)

float sigmoidf(float x);
float rand_float(void);

Matrix create_matrix(size_t rows, size_t cols);
void free_matrix(Matrix m);

void randomize_matrix(Matrix m, float low, float high);
void fill_matrix(Matrix m, float value);
void apply_sigmoid(Matrix m);

void matrix_dot(Matrix dst, Matrix a, Matrix b);
void matrix_sum(Matrix dst, Matrix a);
void print_matrix(Matrix m, const char *name, size_t padding);
Matrix matrix_row(Matrix dst, size_t row);
void matrix_copy(Matrix dst, Matrix src);
void matrix_shuffle_rows(Matrix m);

#endif // _MATRIX_H_
