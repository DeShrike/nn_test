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

void nn_randomize(NN nn, int minimum, int maximum)
{
    for (size_t i = 0; i < nn.count; ++i)
    {
        randomize_matrix(nn.ws[i], minimum, maximum);
        randomize_matrix(nn.bs[i], minimum, maximum);
    }
}

void nn_zero(NN nn)
{
    for (size_t i = 0; i < nn.count; ++i)
    {
        fill_matrix(nn.as[i], 0);
        fill_matrix(nn.ws[i], 0);
        fill_matrix(nn.bs[i], 0);
    }
    
    fill_matrix(nn.as[nn.count], 0);
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

void nn_backprop(NN nn, NN g, Matrix ti, Matrix to)
{
    assert(ti.rows == to.rows);
    assert(NN_OUTPUT(nn).cols == to.cols);

    nn_zero(g);

    // i = current sample
    // l = current layer
    // j = curent activation 
    // k = previous activation
    size_t n = ti.rows;
    for (size_t i = 0; i < n; ++i)  // for each sample in our input
    {
        matrix_copy(NN_INPUT(nn), matrix_row(ti, i)); // copy input row to input layer of network
        nn_forward(nn);
 
        for (size_t j = 0; j <= nn.count; ++j)
        {
            fill_matrix(g.as[j], 0);
        }
 
        for (size_t j = 0; j < to.cols; ++j)    // for each output column
        {
            MATRIX_AT(NN_OUTPUT(g), 0, j) = 
                MATRIX_AT(NN_OUTPUT(nn), 0, j) - MATRIX_AT(to, i, j);    // assumes only 1 output node
        }

        for (size_t l = nn.count; l > 0; --l)  // for each layer
        {
            for (size_t j = 0; j < nn.as[l].cols; ++j)
            {
                float a = MATRIX_AT(nn.as[l], 0, j);    // activation 
                float da = MATRIX_AT(g.as[l], 0, j);    // derivative of activation
                MATRIX_AT(g.bs[l - 1], 0, j) += 2 * da * a * (1 - a);
                for (size_t k = 0; k < nn.as[l - 1].cols; ++k)          // for each previous activation (to the right)
                {
                    // j - weight matrix cols
                    // k - weight matrix row
                    float pa = MATRIX_AT(nn.as[l - 1], 0, k);
                    float w = MATRIX_AT(nn.ws[l - 1], k, j);
                    MATRIX_AT(g.ws[l - 1], k, j) += 2 * da * a * (1 - a) * pa;
                    MATRIX_AT(g.as[l - 1], 0, k) += 2 * da * a * (1 - a) * w;
                }
            }
        }
    }

    for (size_t i = 0; i < g.count; ++i)
    {
        for (size_t j = 0; j < g.ws[i].rows; ++j)
        {
            for (size_t k = 0; k < g.ws[i].cols; ++k)
            {
                MATRIX_AT(g.ws[i], j, k) /= n;
            }
        }

        for (size_t j = 0; j < g.bs[i].rows; ++j)
        {
            for (size_t k = 0; k < g.bs[i].cols; ++k)
            {
                MATRIX_AT(g.bs[i], j, k) /= n;
            }
        }
    }
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
