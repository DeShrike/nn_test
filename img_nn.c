#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "image.h"

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

	write_png_file(OUTPUT_FILENAME, i);

	free_image(i);
	return 0;	
}
