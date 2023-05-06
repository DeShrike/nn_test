#include <stdio.h>
#include "matrix.h"

float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

Matrix create_matrix(size_t rows, size_t cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = malloc(sizeof(*m.data) * rows * cols);
    assert(m.data != NULL);
    return m;
}

void free_matrix(Matrix m)
{
    free(m.data);
}

float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

void randomize_matrix(Matrix m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            MATRIX_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void fill_matrix(Matrix m, float value)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            MATRIX_AT(m, i, j) = value;
        }
    }
}

void apply_sigmoid(Matrix m)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            MATRIX_AT(m, i, j) = sigmoidf(MATRIX_AT(m, i, j));
        }
    }
}

void matrix_dot(Matrix dst, Matrix a, Matrix b)
{
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);

    size_t n = a.cols;
    for (size_t i = 0; i < dst.rows; ++i)
    {
        for (size_t j = 0; j < dst.cols; ++j)
        {
            MATRIX_AT(dst, i, j) = 0;
            for (size_t k = 0; k < n; ++k)
            {
                MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, k) * MATRIX_AT(b, k, j);
            }
        }
    }
}

void matrix_sum(Matrix dst, Matrix a)
{
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; ++i)
    {
        for (size_t j = 0; j < dst.cols; ++j)
        {
            MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, j);
        }
    }
}

void print_matrix(Matrix m, const char *name, size_t padding)
{
    printf("%*s", (int)padding, "");
    printf("%s = [\n", name);
    for (size_t i = 0; i < m.rows; ++i)
    {
        printf("%*s", (int)padding, "");
        for (size_t j = 0; j < m.cols; ++j)
        {
            printf("  %f ", MATRIX_AT(m, i, j));
        }

        printf("\n");
    }

    printf("%*s", (int)padding, "");
    printf("]\n");
}

Matrix matrix_row(Matrix m, size_t row)
{
    return (Matrix){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .data = &MATRIX_AT(m, row, 0)};
}

void matrix_copy(Matrix dst, Matrix src)
{
    // printf("Copy (%ld, %ld) -> (%ld, %ld)\n", src.rows, src.cols, dst.rows, dst.cols);
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; ++i)
    {
        for (size_t j = 0; j < dst.cols; ++j)
        {
            MATRIX_AT(dst, i, j) = MATRIX_AT(src, i, j);
        }
    }
}
