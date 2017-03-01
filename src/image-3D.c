#include "image-3D.h"
#include "utils.h"
#include "blas.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

image make_empty_image(int w, int h, int d, int c)
{
    image out;
    out.data = 0;
	out.d = d;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int d, int c)
{
    image out = make_empty_image(w,h,d,c);
    out.data = calloc(d*h*w*c, sizeof(float));
   return out;
}

void free_image(image m)
{
    free(m.data);
}
