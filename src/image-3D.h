/******************************************************************************
* filename: 
*******************************************************************************/
#ifndef IMAGE_3D_H
#define IMAGE_3D_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>

typedef struct {
	int d;
    int h;
    int w;
    int c;
    float *data;
} image;

image make_image(int w, int h, int d, int c);
image make_empty_image(int w, int h, int d, int c);

void free_image(image m);
#endif

