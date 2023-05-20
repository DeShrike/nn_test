#ifndef _IMAGE_H_
#define _IMAGE_H_

#include "png.h"

typedef struct 
{
	int width, height;
	png_byte color_type;
	png_byte bit_depth;
	png_bytep *row_pointers;
} Image;

Image alloc_image(int width, int height);
void free_image(Image image);
Image read_png_file(char *filename);
void write_png_file(char *filename, Image image);

// printf("PNG_COLOR_TYPE_GRAY: %d\n", PNG_COLOR_TYPE_GRAY);
// printf("PNG_COLOR_TYPE_PALETTE: %d\n", PNG_COLOR_TYPE_PALETTE);
// printf("PNG_COLOR_TYPE_RGB: %d\n", PNG_COLOR_TYPE_RGB);
// printf("PNG_COLOR_TYPE_RGBA: %d\n", PNG_COLOR_TYPE_RGBA);
// printf("PNG_COLOR_TYPE_GA: %d\n", PNG_COLOR_TYPE_GA);

#endif
