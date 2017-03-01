/******************************************************************************
* filename: 
*******************************************************************************/
#include "avgpool_layer.h"
#include "cuda.h"
#include <stdio.h>
#ifndef DEBUG
//#define DEBUG
#endif
avgpool_layer make_avgpool_layer(int batch, int w, int h, int d, int c)
{
    fprintf(stderr, "Avgpool Layer: %d x %d x %d x %d image\n", w,h,d,c);
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
	l.d = d;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
	l.out_d = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = d*h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
#ifdef GPU
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
#endif
    return l;
}

void forward_avgpool_layer(const avgpool_layer l, network_state state)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.d*l.h*l.w; ++i){
                int in_index = i + l.d*l.h*l.w*(k + b*l.c);
                l.output[out_index] += state.input[in_index];
            }
            l.output[out_index] /= l.d*l.h*l.w;
        }
    }
}

void backward_avgpool_layer(const avgpool_layer l, network_state state)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.d*l.h*l.w; ++i){
                int in_index = i + l.d*l.h*l.w*(k + b*l.c);
                state.delta[in_index] += l.delta[out_index] / (l.d*l.h*l.w);
            }
        }
    }
}
