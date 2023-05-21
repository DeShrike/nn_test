#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include "image.h"
#include "nn.h"

#define MAX_EPOCHS      500 * 1000
#define LEARNING_RATE   1

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
                out_image.color_type = PNG_COLOR_TYPE_PALETTE;
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

Image process_color(Image image)
{
    bool is_color = image.color_type == PNG_COLOR_TYPE_RGB || image.color_type == PNG_COLOR_TYPE_RGBA;
    int output_cols = is_color ? 3 : 1;

    Matrix t = create_matrix(image.width * image.height, 2 + 3);
    for (int y = 0; y < image.height; ++y)
    {
        for (int x = 0; x < image.width; ++x)
        {
            size_t i = y * image.width + x;
			uint r = image.row_pointers[y][x * 4 + 0];
			uint g = image.row_pointers[y][x * 4 + 1];
			uint b = image.row_pointers[y][x * 4 + 2];
            MATRIX_AT(t, i, 0) = (float)x / (image.width - 1);
            MATRIX_AT(t, i, 1) = (float)y / (image.height - 1);
            MATRIX_AT(t, i, 2) = r / 255.f;
            MATRIX_AT(t, i, 3) = g / 255.f;
            MATRIX_AT(t, i, 4) = b / 255.f;
        }
    }

    matrix_shuffle_rows(t);

    Matrix ti = {
        .rows = t.rows,
        .cols = 2,
        .stride = t.stride,
        .data = &MATRIX_AT(t, 0, 0),
    };
    Matrix to = {
        .rows = t.rows,
        .cols = 3,
        .stride = t.stride,
        .data = &MATRIX_AT(t, 0, ti.cols),
    };

    // MATRIX_PRINT(ti);
    // MATRIX_PRINT(to);

    size_t arch[] = {2, image.width / 1, image.width / 2, 3};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));
    nn_randomize(nn, -1, 1);

    float rate = 1;
    size_t max_epochs = 500 * 1000;
    size_t use_stochastic_gradient_descent = 1;

    size_t batch_size = image.width;
    size_t batch_count = (t.rows + batch_size - 1) / batch_size;

    float c = nn_cost(nn, ti, to);
    float cc = c;
    float diff = 0;

    for (size_t epoch = 0; epoch < max_epochs; ++epoch)
    {
       if (use_stochastic_gradient_descent == 1)
        {
            size_t batch_current = epoch % batch_count;
            Matrix batch_ti = {
                .rows = batch_size,
                .cols = 2,
                .stride = t.stride,
                .data = &MATRIX_AT(t, batch_current * batch_size, 0),
            };
            Matrix batch_to = {
                .rows = batch_size,
                .cols = 3,
                .stride = t.stride,
                .data = &MATRIX_AT(t, batch_current * batch_size, batch_ti.cols),
            };

            nn_backprop(nn, gradient, batch_ti, batch_to);
        }
        else
        {
            nn_backprop(nn, gradient, ti, to);
        }

        nn_learn(nn, gradient, rate);
        if (epoch % 100 == 0)
        {
            c = nn_cost(nn, ti, to);
            diff = c - cc;
            cc = c;
            printf("%zu: Cost = %f (%f)  \r", epoch, c, diff);
            fflush(stdout);
            if (c < 0.001)
            {
                break;
            }
        }
    }

    printf("\n");

    Image out_image = upscale(image, nn);

    return out_image;
}

Image process_gray(Image image)
{
    if (image.width <= 28)
    {
        for (int y = 0; y < image.height; ++y)
        {
            for (int x = 0; x < image.width; ++x)
            {
                uint pixel = image.row_pointers[y][x * 4];
                printf("%3u ", pixel);
            }

            printf("\n");
        }
    }

    Matrix t = create_matrix(image.width * image.height, 2 + 1);
    for (int y = 0; y < image.height; ++y)
    {
        for (int x = 0; x < image.width; ++x)
        {
            size_t i = y * image.width + x;
			uint pixel = image.row_pointers[y][x * 4];
            MATRIX_AT(t, i, 0) = (float)x / (image.width - 1);
            MATRIX_AT(t, i, 1) = (float)y / (image.height - 1);
            MATRIX_AT(t, i, 2) = pixel / 255.f;
        }
    }

    matrix_shuffle_rows(t);

    Matrix ti = {
        .rows = t.rows,
        .cols = 2,
        .stride = t.stride,
        .data = &MATRIX_AT(t, 0, 0),
    };
    Matrix to = {
        .rows = t.rows,
        .cols = 1,
        .stride = t.stride,
        .data = &MATRIX_AT(t, 0, ti.cols),
    };

    // MATRIX_PRINT(ti);
    // MATRIX_PRINT(to);

    size_t arch[] = {2, image.width / 2, image.width / 4, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));
    nn_randomize(nn, -1, 1);

    size_t use_stochastic_gradient_descent = 1;
    float rate = 1;
    size_t max_epochs = 500 * 1000;
    size_t batch_size = image.width;
    size_t batch_count = (t.rows + batch_size - 1) / batch_size;
    float c = nn_cost(nn, ti, to);
    float cc = c;
    float diff = 0;

    printf("batch_count: %zu, batch_size: %zu\n", batch_count, batch_size);

    for (size_t epoch = 0; epoch < max_epochs; ++epoch)
    {
        if (use_stochastic_gradient_descent == 1)
        {
            size_t batch_current = epoch % batch_count;
            Matrix batch_ti = {
                .rows = batch_size,
                .cols = 2,
                .stride = t.stride,
                .data = &MATRIX_AT(t, batch_current * batch_size, 0),
            };
            Matrix batch_to = {
                .rows = batch_size,
                .cols = 1,
                .stride = t.stride,
                .data = &MATRIX_AT(t, batch_current * batch_size, batch_ti.cols),
            };

            nn_backprop(nn, gradient, batch_ti, batch_to);
        }
        else
        {
            nn_backprop(nn, gradient, ti, to);
        }

        nn_learn(nn, gradient, rate);

        if (epoch % 100 == 0)
        {
            c = nn_cost(nn, ti, to);
            diff = c - cc;
            cc = c;
            printf("%zu: Cost = %f (%f)  \r", epoch, c, diff);
            fflush(stdout);
            if (c < 0.001)
            {
                break;
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

    // printf("Dimensions: %d x %d\n", i.width, i.height);
    // printf("Color Type: %d\n", i.color_type);
    // printf("Bit Depth: %d\n", i.bit_depth);

	if (i.bit_depth != 8)
	{
		fprintf(stderr, "Only 8 bit images are supported\n");
		abort();
	}

	if (i.color_type == PNG_COLOR_TYPE_PALETTE)  
	{
    	Image o = process_gray(i);
        printf("Saving %s\n", output_file_path);
        write_png_file(output_file_path, o);
        free_image(o);
	}
    else if (i.color_type == PNG_COLOR_TYPE_RGB || i.color_type == PNG_COLOR_TYPE_RGBA)
    {
    	Image o = process_color(i);
        printf("Saving %s\n", output_file_path);
        write_png_file(output_file_path, o);
        free_image(o);
    }

	free_image(i);
	return 0;	
}
