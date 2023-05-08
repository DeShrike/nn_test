#include <stdio.h>
#include <math.h>
#include <time.h>
#include "nn.h"

typedef struct
{
    int rows;
    int input_cols;
    int output_cols;
    int stride;
    float training_data[100];
} Model;

Model xor_model = {
    .input_cols = 2,
    .output_cols = 1,
    .stride = 3,
    .training_data = {
        0, 0, 0, // input 1, intput 2, expected
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    },
    .rows = 4,
};

int main(void)
{
    srand(time(NULL));

    Model model = xor_model;

    Matrix ti = {
        .rows = model.rows,
        .cols = model.input_cols,
        .stride = model.stride,
        .data = model.training_data,
    };

    Matrix to = {
        .rows = model.rows,
        .cols = model.output_cols,
        .stride = model.stride,
        .data = model.training_data + model.input_cols,
    };

    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));
    nn_randomize(nn);

    float rate = 1;

    float c = nn_cost(nn, ti, to);
    printf("Cost = %f\n", c);

    size_t it;
    for (it = 0; it < 100 * 1000; ++it)
    {
#if 0
        float eps = 1e-1;
        nn_finite_diff(nn, gradient, eps, ti, to);
#else
        nn_backprop(nn, gradient, ti, to);
#endif
        nn_learn(nn, gradient, rate);
        c = nn_cost(nn, ti, to);
        if (c < 0.0001)
        {
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
