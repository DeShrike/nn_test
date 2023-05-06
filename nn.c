#include <stdio.h>
#include "nn.h"

// size_t arch[] = { 2, 2, 1 }; // input-, hidden-, output-nodes
// NN nn = nn_alloc(arch, ARRAY_LEN(arch));

NN nn_alloc(size_t *arch, size_t arch_count)
{
    assert(arch_count > 0);
    NN nn;
    nn.count = arch_count - 1;

    nn.ws = malloc(sizeof(*nn.ws) * nn.count);
    assert(nn.ws != NULL);
    nn.bs = malloc(sizeof(*nn.bs) * nn.count);
    assert(nn.bs != NULL);
    nn.as = malloc(sizeof(*nn.as) * (nn.count + 1));
    assert(nn.as != NULL);

    nn.as[0] = create_matrix(1, arch[0]);
    for (size_t i = 1; i < arch_count; ++i)
    {
        nn.ws[i - 1] = create_matrix(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = create_matrix(1, arch[i]);
        nn.as[i] = create_matrix(1, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char *name)
{
    char buf[100];
    printf("%s = [\n", name);
    Matrix *ws = nn.ws;
    Matrix *bs = nn.bs;
    for (size_t i = 0; i < nn.count; ++i)
    {
        snprintf(buf, sizeof(buf), "ws %zu", i + 1);
        print_matrix(ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs %zu", i + 1);
        print_matrix(bs[i], buf, 4);
    }

    printf("]\n");
}

void nn_free(NN nn)
{
    free_matrix(nn.as[0]);

    for (size_t i = 1; i < nn.count + 1; ++i)
    {
        free_matrix(nn.ws[i - 1]);
        free_matrix(nn.bs[i - 1]);
        free_matrix(nn.as[i]);
    }

    free(nn.as);
    free(nn.bs);
    free(nn.ws);
}

void nn_randomize(NN nn)
{
    for (size_t i = 0; i < nn.count; ++i)
    {
        randomize_matrix(nn.ws[i], 0, 1);
        randomize_matrix(nn.bs[i], 0, 1);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; ++i)
    {
        matrix_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        matrix_sum(nn.as[i + 1], nn.bs[i]); // add bias
        apply_sigmoid(nn.as[i + 1]);
    }
}

float nn_cost(NN nn, Matrix ti, Matrix to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;
    float c = 0;

    for (size_t i = 0; i < n; ++i)
    {
        Matrix x = matrix_row(ti, i);
        Matrix y = matrix_row(to, i);

        matrix_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        size_t q = to.cols;
        for (size_t j = 0; j < q; ++j)
        {
            float diff = MATRIX_AT(NN_OUTPUT(nn), 0, j) - MATRIX_AT(y, 0, j);
            c += diff * diff;
        }
    }

    return c / n;
}

void nn_finite_diff(NN nn, NN g, float eps, Matrix ti, Matrix to)
{
    float saved;
    float c = nn_cost(nn, ti, to);
    for (size_t i = 0; i < nn.count; ++i)
    {
        for (size_t j = 0; j < nn.ws[i].rows; ++j)
        {
            for (size_t k = 0; k < nn.ws[i].cols; ++k)
            {
                saved = MATRIX_AT(nn.ws[i], j, k);
                MATRIX_AT(nn.ws[i], j, k) += eps;
                MATRIX_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MATRIX_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j)
        {
            for (size_t k = 0; k < nn.bs[i].cols; ++k)
            {
                saved = MATRIX_AT(nn.bs[i], j, k);
                MATRIX_AT(nn.bs[i], j, k) += eps;
                MATRIX_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MATRIX_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nn_learn(NN nn, NN g, float rate)
{
    for (size_t i = 0; i < nn.count; ++i)
    {
        for (size_t j = 0; j < nn.ws[i].rows; ++j)
        {
            for (size_t k = 0; k < nn.ws[i].cols; ++k)
            {
                MATRIX_AT(nn.ws[i], j, k) -= rate * MATRIX_AT(g.ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j)
        {
            for (size_t k = 0; k < nn.bs[i].cols; ++k)
            {
                MATRIX_AT(nn.bs[i], j, k) -= rate * MATRIX_AT(g.bs[i], j, k);
            }
        }
    }
}
