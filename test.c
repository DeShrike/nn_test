#include <stdio.h>
#include <math.h>
#include <time.h>
#include "nn.h"

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

    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));
    nn_randomize(nn);

    float eps = 1e-1;
    float rate = 1e-1;

    float c = nn_cost(nn, ti, to);
    printf("Cost = %f\n", c);

    size_t it;
    for (it = 0; it < 100 * 1000; ++it)
    {
        nn_finite_diff(nn, gradient, eps, ti, to);
        nn_learn(nn, gradient, rate);
        c = nn_cost(nn, ti, to);
        if (c < 0.001)
        {
            printf("%zu\n", it);
            break;
        }
    }

    printf("---------------------------------------\n");
    NN_PRINT(nn);
    printf("---------------------------------------\n");
    printf("Cost = %f (after %zu iterations)\n", c, it);
    printf("---------------------------------------\n");
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            MATRIX_AT(NN_INPUT(nn), 0, 0) = i;
            MATRIX_AT(NN_INPUT(nn), 0, 1) = j;

            nn_forward(nn);
            float y = MATRIX_AT(NN_OUTPUT(nn), 0, 0);

            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }

    nn_free(gradient);
    nn_free(nn);

    return 0;
}
