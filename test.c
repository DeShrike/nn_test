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

Model adder_model = {
    .input_cols = 4,
    .output_cols = 2,
    .stride = 6,
    .training_data = {
        0, 0,  0, 0,  0, 0,
        0, 0,  0, 1,  0, 1,
        0, 0,  1, 0,  1, 0,
        0, 0,  1, 1,  1, 1,
        0, 1,  0, 0,  0, 1,
        0, 1,  0, 1,  1, 0,
        0, 1,  1, 0,  1, 1,
        0, 1,  1, 1,  0, 0,
        1, 0,  0, 0,  1, 0,
        1, 0,  0, 1,  1, 1,
        1, 0,  1, 0,  0, 0,
        1, 0,  1, 1,  0, 1,
        1, 1,  0, 0,  1, 1,
        1, 1,  0, 1,  0, 0,
        1, 1,  1, 0,  0, 1,
        1, 1,  1, 1,  1, 0,
    },
    .rows = 16,
};

/*
float training_data[] = {
    0, 0, 0, // input 1, intput 2, expected
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};
*/
int main(void)
{
    srand(time(NULL));

    Model model = xor_model;
    // Model model = adder_model;

    // size_t rows = sizeof(training_data) / sizeof(training_data[0]) / model.stride;
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

    // MATRIX_PRINT(to);

    // return 0;
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
            break;
        }
    }

    printf("---------------------------------------\n");
    NN_PRINT(nn);
    printf("---------------------------------------\n");
    printf("Cost = %f (after %zu iterations)\n", c, it);
    printf("---------------------------------------\n");

    /*
    for (int i = 0; i < 16; ++i)
    {
        size_t a1 = (i & 1) == 1 ? 1 : 0;
        size_t a2 = (i & 2) == 2 ? 1 : 0;
        size_t a3 = (i & 4) == 4 ? 1 : 0;
        size_t a4 = (i & 8) == 8 ? 1 : 0;

        MATRIX_AT(NN_INPUT(nn), 0, 0) = a1;
        MATRIX_AT(NN_INPUT(nn), 0, 1) = a2;
        MATRIX_AT(NN_INPUT(nn), 0, 2) = a3;
        MATRIX_AT(NN_INPUT(nn), 0, 3) = a4;
        nn_forward(nn);

        float y1 = MATRIX_AT(NN_OUTPUT(nn), 0, 0);
        float y2 = MATRIX_AT(NN_OUTPUT(nn), 0, 1);

        printf("%zu %zu & %zu %zu = %f %f\n", a1, a2, a3, a4, y1, y2);
    }
    */

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
