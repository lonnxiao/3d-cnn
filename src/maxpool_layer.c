/******************************************************************************
* filename: 
* file function: 
* last modified: 
*******************************************************************************/
#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

maxpool_layer make_maxpool_layer(int batch, int d, int h, int w, int c, int size, int stride)
{
    fprintf(stderr, "Maxpool Layer: %d x %d x %d x %d image, %d size, %d stride\n", d,h,w,c,size,stride);
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.d = d;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = (w-1)/stride + 1;
    l.out_h = (h-1)/stride + 1;
    l.out_d = (d-1)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_d * l.out_h * l.out_w * l.out_c;
    l.inputs = d*h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_d * l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
#ifdef GPU
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
#endif
    return l;
}

void forward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int b,i,j,k,m,n,o,s;
    int w_offset = (-l.size-1)/2 + 1;
    int h_offset = (-l.size-1)/2 + 1;
    int d_offset = (-l.size-1)/2 + 1;

    int d = (l.d-1)/l.stride + 1;
    int h = (l.h-1)/l.stride + 1;
    int w = (l.w-1)/l.stride + 1;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
			for(s = 0; s < d; ++s){
				for(i = 0; i < h; ++i){
					for(j = 0; j < w; ++j){
						int out_index = j + w*(i + h*(s + d*(k + c*b)));
						float max = -FLT_MAX;
						int max_i = -1;
						for(o = 0; o < l.size; ++o){
							for(n = 0; n < l.size; ++n){
								for(m = 0; m < l.size; ++m){
									int cur_d = d_offset + s*l.stride + o;
									int cur_h = h_offset + i*l.stride + n;
									int cur_w = w_offset + j*l.stride + m;
									int index = cur_w + l.w*(cur_h + l.h*(cur_d + l.d*(k + b*l.c)));
									int valid = (cur_d >= 0 && cur_d < l.d &&
												 cur_h >= 0 && cur_h < l.h &&
												 cur_w >= 0 && cur_w < l.w);
									float val = (valid != 0) ? state.input[index] : -FLT_MAX;
									max_i = (val > max) ? index : max_i;
									max   = (val > max) ? val   : max;
								}
							}
						}
						l.output[out_index] = max;
						l.indexes[out_index] = max_i;
					}
				}
			}
		}
    }
}

void backward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int i;
    int d = (l.d-1)/l.stride + 1;
    int h = (l.h-1)/l.stride + 1;
    int w = (l.w-1)/l.stride + 1;
    int c = l.c;
    for(i = 0; i < d*h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        state.delta[index] += l.delta[i];
    }
}

