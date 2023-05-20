#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "image.h"
#include "nn.h"

#define INPUT_FILENAME "8.png"
#define OUTPUT_FILENAME "upscaled.png"

void process(Image image)
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

    Matrix t = create_matrix(image.width * image.height, 3);
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

    size_t arch[] = {2, 7, 7, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));
    nn_randomize(nn, -1, 1);

    float rate = 1;
    float c = nn_cost(nn, ti, to);
    printf("Cost = %f\n", c);

    size_t max_epochs = 200 * 1000;
    for (size_t i = 0; i < max_epochs; ++i)
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

    size_t out_width = 512;
    size_t out_height = 512;

    printf("Allocing\n");
    Image out_image = alloc_image(out_width, out_height);
    
    printf("Creating\n");
    for (size_t y = 0; y < out_height; ++y)
    {
        for (size_t x = 0; x < out_width; ++x)
        {
            MATRIX_AT(NN_INPUT(nn), 0, 0) = (float)x / (out_width - 1);
            MATRIX_AT(NN_INPUT(nn), 0, 1) = (float)y / (out_height - 1);
            nn_forward(nn);
            uint8_t pixel = MATRIX_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
			out_image.row_pointers[y][x * 4 + 0] = pixel;
			out_image.row_pointers[y][x * 4 + 1] = pixel;
			out_image.row_pointers[y][x * 4 + 2] = pixel;
			out_image.row_pointers[y][x * 4 + 3] = 255;
        }
    }

    printf("Saving\n");
    write_png_file(OUTPUT_FILENAME, out_image);

    printf("Freeing\n");
	free_image(out_image);
}

int main()
{
	Image i = read_png_file(INPUT_FILENAME);

	if (i.bit_depth != 8 || i.color_type != 3)
	{
		fprintf(stderr, "Only 8 bit monochrome images are supported\n");
		abort();
	}

	process(i);

	free_image(i);
	return 0;	
}
