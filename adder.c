#include <stdio.h>
#include <math.h>
#include <time.h>
#include "nn.h"

#define BITS 4

int main()
{
    srand(time(NULL));

    size_t n = (1 << BITS);
    size_t rows = n * n;
    Matrix ti = create_matrix(rows, 2 * BITS);
    Matrix to = create_matrix(rows, BITS + 1); // + carry bit
    for (size_t i = 0; i < ti.rows; ++i)
    {
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y;
        size_t overflow = z >= n;
        for (size_t j = 0; j < BITS; ++j)
        {
            MATRIX_AT(ti, i, j) = (x >> j) & 1;
            MATRIX_AT(ti, i, j + BITS) = (y >> j) & 1;
            if (overflow)
            {
                MATRIX_AT(to, i, j)        = 0;
            }
            else
            {
                MATRIX_AT(to, i, j) = (z >> j) & 1;
            }
        }

        MATRIX_AT(to, i, BITS) = overflow;
    }

    // MATRIX_PRINT(ti);
    // MATRIX_PRINT(to);

    size_t arch[] = { 2 * BITS, BITS * 5, BITS + 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));
    nn_randomize(nn);

    float rate = 1;
    float c = nn_cost(nn, ti, to);
    printf("Cost = %f\n", c);

    for (size_t i = 0; i < 100 * 1000; ++i)
    {
        nn_backprop(nn, gradient, ti, to);
        nn_learn(nn, gradient, rate);
        if (i % 100 == 0)
        {
            c = nn_cost(nn, ti, to);
            printf("%zu: Cost = %f\n", i, c);
            if (c < 0.005)
            {
                break;
            }
        }
    }

    NN_PRINT(nn);

    // Verify
    int good_count = 0;
    int bad_count = 0;
    for (size_t x = 0; x < n; ++x)
    {
        for (size_t y = 0; y < n; ++y)
        {
            for (size_t j = 0; j < BITS; ++j)
            {
                MATRIX_AT(NN_INPUT(nn), 0, j) = (x >> j) & 1;
                MATRIX_AT(NN_INPUT(nn), 0, j + BITS) = (y >> j) & 1;
            }

            nn_forward(nn);

            if (MATRIX_AT(NN_OUTPUT(nn), 0, BITS) > 0.5)
            {
                if (x + y >= n)
                {
                    good_count++;
                }
                else
                {
                    bad_count++;
                    printf("%zu + %zu = " , x, y);
                    printf("OVERFLOW\n");
                }
            }
            else
            {
                size_t z = 0;
                for (size_t j = 0; j < BITS; ++j)
                {
                    size_t bit = MATRIX_AT(NN_OUTPUT(nn), 0, j) > 0.5;
                    z |= (bit << j);
                }

                if (x + y == z)
                {
                    good_count++;
                }
                else
                {
                    bad_count++;
                    printf("%zu + %zu = " , x, y);
                    printf("%zu\n", z);
                }
            }
        }
    }

    printf("Correct: %d - Wrong: %d\n", good_count, bad_count);

    nn_free(gradient);
    nn_free(nn);
    return 0;
}
