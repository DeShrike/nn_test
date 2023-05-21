#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include "image.h"
#include "nn.h"

#define MAX_EPOCHS                          1000 * 1000
#define LEARNING_RATE                       1
#define USE_STOCHASTIC_GRADIENT_DESCENT     true
#define MINIMUM_COST                        0.005

char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

Image upscale(Image input, NN nn)
{
    size_t out_width = input.width * 10;
    size_t out_height = input.height * 10;
    
    printf("Creating %zu x %zu output image\n", out_width, out_height);
    Image out_image = alloc_image(out_width, out_height);

    for (size_t y = 0; y < out_height; ++y)
    {
        for (size_t x = 0; x < out_width; ++x)
        {
            MATRIX_AT(NN_INPUT(nn), 0, 0) = (float)x / (out_width - 1);
            MATRIX_AT(NN_INPUT(nn), 0, 1) = (float)y / (out_height - 1);
            nn_forward(nn);
            if (input.color_type == PNG_COLOR_TYPE_PALETTE)
            {
                out_image.color_type = PNG_COLOR_TYPE_PALETTE;
                uint8_t pixel = MATRIX_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
                out_image.row_pointers[y][x * 4 + 0] = pixel;
                out_image.row_pointers[y][x * 4 + 1] = pixel;
                out_image.row_pointers[y][x * 4 + 2] = pixel;
                out_image.row_pointers[y][x * 4 + 3] = 255;
            }
            else if (input.color_type == PNG_COLOR_TYPE_RGB || input.color_type == PNG_COLOR_TYPE_RGBA)
            {
                out_image.color_type = PNG_COLOR_TYPE_RGBA;
                uint8_t r = MATRIX_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
                uint8_t g = MATRIX_AT(NN_OUTPUT(nn), 0, 1) * 255.f;
                uint8_t b = MATRIX_AT(NN_OUTPUT(nn), 0, 2) * 255.f;
                out_image.row_pointers[y][x * 4 + 0] = r;
                out_image.row_pointers[y][x * 4 + 1] = g;
                out_image.row_pointers[y][x * 4 + 2] = b;
                out_image.row_pointers[y][x * 4 + 3] = 255;
            }
        }
    }

    return out_image;
}

Image process(Image image)
{
    bool is_color = (image.color_type == PNG_COLOR_TYPE_RGB) || (image.color_type == PNG_COLOR_TYPE_RGBA);
    int output_cols = is_color ? 3 : 1;

    Matrix t = create_matrix(image.width * image.height, 2 + output_cols);
    for (int y = 0; y < image.height; ++y)
    {
        for (int x = 0; x < image.width; ++x)
        {
            size_t i = y * image.width + x;
            if (is_color)
            {
                uint r = image.row_pointers[y][x * 4 + 0];
                uint g = image.row_pointers[y][x * 4 + 1];
                uint b = image.row_pointers[y][x * 4 + 2];
                MATRIX_AT(t, i, 0) = (float)x / (image.width - 1);
                MATRIX_AT(t, i, 1) = (float)y / (image.height - 1);
                MATRIX_AT(t, i, 2) = r / 255.f;
                MATRIX_AT(t, i, 3) = g / 255.f;
                MATRIX_AT(t, i, 4) = b / 255.f;
            }
            else
            {
                uint pixel = image.row_pointers[y][x * 4];
                MATRIX_AT(t, i, 0) = (float)x / (image.width - 1);
                MATRIX_AT(t, i, 1) = (float)y / (image.height - 1);
                MATRIX_AT(t, i, 2) = pixel / 255.f;
            }
        }
    }

    Matrix ti = {
        .rows = t.rows,
        .cols = 2,
        .stride = t.stride,
        .data = &MATRIX_AT(t, 0, 0),
    };
    Matrix to = {
        .rows = t.rows,
        .cols = output_cols,
        .stride = t.stride,
        .data = &MATRIX_AT(t, 0, ti.cols),
    };

    // MATRIX_PRINT(ti);
    // MATRIX_PRINT(to);

    size_t arch[] = { 2, image.width / 1, image.width / 2, output_cols };
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));
    nn_randomize(nn, -1, 1);

    float rate = 1;
    size_t max_epochs = MAX_EPOCHS;
    bool use_stochastic_gradient_descent = USE_STOCHASTIC_GRADIENT_DESCENT;

    float cost = nn_cost(nn, ti, to);
    float prev_cost = cost;
    float cost_diff = 0;

    if (use_stochastic_gradient_descent == false)
    {
        matrix_shuffle_rows(t);
        for (size_t epoch = 0; epoch < max_epochs; ++epoch)
        {
            nn_backprop(nn, gradient, ti, to);
            nn_learn(nn, gradient, rate);

            if (epoch % 100 == 0)
            {
                cost = nn_cost(nn, ti, to);
                cost_diff = cost - prev_cost;
                prev_cost = cost;
                printf("%zu: Cost = %f (%f)  \r", epoch, cost, cost_diff);
                fflush(stdout);
                if (cost < MINIMUM_COST)
                {
                    break;
                }
            }
        }
    }
    else
    {
        size_t batch_size = image.width;
        size_t batch_count = (t.rows + batch_size - 1) / batch_size;

        printf("Batch Count: %zu - Batch Size: %zu\n", batch_count, batch_size);

        cost = 0.0f;
        size_t batch_current = 0;
        for (size_t epoch = 0; epoch < max_epochs; ++epoch)
        {
            Matrix batch_ti = {
                .rows = batch_size,
                .cols = 2,
                .stride = t.stride,
                .data = &MATRIX_AT(t, batch_current * batch_size, 0),
            };
            Matrix batch_to = {
                .rows = batch_size,
                .cols = output_cols,
                .stride = t.stride,
                .data = &MATRIX_AT(t, batch_current * batch_size, batch_ti.cols),
            };

            nn_backprop(nn, gradient, batch_ti, batch_to);
            nn_learn(nn, gradient, rate);

            cost += nn_cost(nn, batch_ti, batch_to);
            if (batch_current == batch_count - 1)
            {
                cost = cost / batch_count;

                if (epoch % (batch_count * 5 - 1) == 0)
                {
                    cost_diff = cost - prev_cost;
                    prev_cost = cost;
                    printf("%zu: Cost = %f (%f)  \r", epoch, cost, cost_diff);
                    fflush(stdout);
                    if (cost < MINIMUM_COST)
                    {
                        break;
                    }
                }

                cost = 0.0f;
                batch_current = 0;
                matrix_shuffle_rows(t);
            }
            else
            {
                batch_current += 1;
            }
        }
    }

    printf("\n");

    Image out_image = upscale(image, nn);

    return out_image;
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    const char *program = args_shift(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <input> <output>\n", program);
        fprintf(stderr, "ERROR: no input file is provided\n");
        return 1;
    }

    const char *input_file_path = args_shift(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <input> <output>\n", program);
        fprintf(stderr, "ERROR: no output file is provided\n");
        return 1;
    }

    const char *output_file_path = args_shift(&argc, &argv);

    printf("Reading %s\n", input_file_path);
	Image i = read_png_file(input_file_path);
    printf("Dimensions: %d x %d\n", i.width, i.height);

    // printf("Color Type: %d\n", i.color_type);
    // printf("Bit Depth: %d\n", i.bit_depth);

	if (i.bit_depth != 8)
	{
		fprintf(stderr, "Only 8 bit images are supported\n");
    	free_image(i);
		abort();
	}

    Image o = process(i);
    
    printf("Saving %s\n", output_file_path);
    write_png_file(output_file_path, o);
    free_image(o);

	free_image(i);
	return 0;	
}
