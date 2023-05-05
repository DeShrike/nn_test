#include <stdio.h>
#include <math.h>
#include <time.h>
#include "ml.h"

typedef struct
{
    Matrix a0;

    Matrix w1, b1, a1;
    Matrix w2, b2, a2;
} Xor;

Xor xor_alloc(void)
{
    Xor m;
    // Input Layer
    m.a0 = create_matrix(1, 2);

    // Layer 1
    m.w1 = create_matrix(2, 2);
    m.b1 = create_matrix(1, 2);

    // Layer 2
    m.a1 = create_matrix(1, 2);

    m.w2 = create_matrix(2, 1);
    m.b2 = create_matrix(1, 1);

    // Output layer
    m.a2 = create_matrix(1, 1);

    return m;
}

void xor_free(Xor m)
{
    free_matrix(m.a0);
    free_matrix(m.a1);
    free_matrix(m.a2);

    free_matrix(m.w1);
    free_matrix(m.w2);
    free_matrix(m.b1);
    free_matrix(m.b2);
}

void forward_xor(Xor m)
{
    matrix_dot(m.a1, m.a0, m.w1);
    matrix_sum(m.a1, m.b1); // add bias
    apply_sigmoid(m.a1);

    matrix_dot(m.a2, m.a1, m.w2);
    matrix_sum(m.a2, m.b2); // add bias
    apply_sigmoid(m.a2);
}

float cost(Xor m, Matrix ti /*training_input*/, Matrix to /*training_ouput*/)
{
    assert(ti.rows == to.rows);
    assert(to.cols == m.a2.cols);
    size_t n = ti.rows;
    float c = 0;
    for (size_t i = 0; i < n; ++i)
    {
        Matrix x = matrix_row(ti, i);
        Matrix y = matrix_row(to, i);

        matrix_copy(m.a0, x);
        forward_xor(m);

        size_t q = to.cols;
        for (size_t j = 0; j < q; ++j)
        {
            float diff = MATRIX_AT(m.a2, 0, j) - MATRIX_AT(y, 0, j);
            c += diff * diff;
        }
    }

    return c / n;
}

void finite_diff(Xor m, Xor g, float eps, Matrix ti, Matrix to)
{
    // calculate gradient
    float saved;
    float c = cost(m, ti, to);

    for (size_t i = 0; i < m.w1.rows; ++i)
    {
        for (size_t j = 0; j < m.w1.cols; ++j)
        {
            saved = MATRIX_AT(m.w1, i, j);
            MATRIX_AT(m.w1, i, j) += eps;
            MATRIX_AT(g.w1, i, j) = (cost(m, ti, to) - c) / eps;
            MATRIX_AT(m.w1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b1.rows; ++i)
    {
        for (size_t j = 0; j < m.b1.cols; ++j)
        {
            saved = MATRIX_AT(m.b1, i, j);
            MATRIX_AT(m.b1, i, j) += eps;
            MATRIX_AT(g.b1, i, j) = (cost(m, ti, to) - c) / eps;
            MATRIX_AT(m.b1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.w2.rows; ++i)
    {
        for (size_t j = 0; j < m.w2.cols; ++j)
        {
            saved = MATRIX_AT(m.w2, i, j);
            MATRIX_AT(m.w2, i, j) += eps;
            MATRIX_AT(g.w2, i, j) = (cost(m, ti, to) - c) / eps;
            MATRIX_AT(m.w2, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b2.rows; ++i)
    {
        for (size_t j = 0; j < m.b2.cols; ++j)
        {
            saved = MATRIX_AT(m.b2, i, j);
            MATRIX_AT(m.b2, i, j) += eps;
            MATRIX_AT(g.b2, i, j) = (cost(m, ti, to) - c) / eps;
            MATRIX_AT(m.b2, i, j) = saved;
        }
    }
}

void xor_learn(Xor m, Xor g, float rate)
{
    // apply gradient
    for (size_t i = 0; i < m.w1.rows; ++i)
    {
        for (size_t j = 0; j < m.w1.cols; ++j)
        {
            MATRIX_AT(m.w1, i, j) -= rate * MATRIX_AT(g.w1, i, j);
        }
    }

    for (size_t i = 0; i < m.b1.rows; ++i)
    {
        for (size_t j = 0; j < m.b1.cols; ++j)
        {
            MATRIX_AT(m.b1, i, j) -= rate * MATRIX_AT(g.b1, i, j);
        }
    }

    for (size_t i = 0; i < m.w2.rows; ++i)
    {
        for (size_t j = 0; j < m.w2.cols; ++j)
        {
            MATRIX_AT(m.w2, i, j) -= rate * MATRIX_AT(g.w2, i, j);
        }
    }

    for (size_t i = 0; i < m.b2.rows; ++i)
    {
        for (size_t j = 0; j < m.b2.cols; ++j)
        {
            MATRIX_AT(m.b2, i, j) -= rate * MATRIX_AT(g.b2, i, j);
        }
    }
}

float training_data[] = {
    0,
    0,
    0, // input 1, intput 2, expected
    0,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    0,
};

int main(void)
{
    srand(time(NULL));

    size_t stride = 3;
    size_t n = sizeof(training_data) / sizeof(training_data[0]) / stride;
    Matrix ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .data = training_data,
    };

    Matrix to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .data = training_data + 2,
    };

    // MATRIX_PRINT(ti);
    // MATRIX_PRINT(to);
    // return 0;

    Xor m = xor_alloc();
    Xor gradient = xor_alloc();

    // XOR

    randomize_matrix(m.w1, 0.0f, 1.0f);
    randomize_matrix(m.b1, 0.0f, 1.0f);

    randomize_matrix(m.w2, 0.0f, 1.0f);
    randomize_matrix(m.b2, 0.0f, 1.0f);

    float eps = 1e-1;
    float rate = 1e-1;

    printf("Cost = %f\n", cost(m, ti, to));
    for (size_t i = 0; i < 30 * 1000; ++i)
    {
        finite_diff(m, gradient, eps, ti, to);
        xor_learn(m, gradient, rate);
        printf("%zu: Cost = %f\n", i, cost(m, ti, to));
    }

#if 1
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            MATRIX_AT(m.a0, 0, 0) = i;
            MATRIX_AT(m.a0, 0, 1) = j;

            forward_xor(m);
            float y = *m.a2.data;

            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }

    // MATRIX_PRINT(m.w1);
    // MATRIX_PRINT(m.b1);
    // MATRIX_PRINT(m.w2);
    // MATRIX_PRINT(m.b2);
#endif

    xor_free(m);
    xor_free(gradient);

    return 0;
}
