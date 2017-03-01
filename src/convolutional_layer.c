/******************************************************************************
* filename:
* file function:
* last modified:
*******************************************************************************/
#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifndef DEBUG
//#define DEBUG
#endif

#ifdef AI2
#include "xnor_layer.h"
#endif

#ifndef AI2
#define AI2 0
void forward_xnor_layer(layer l, network_state state);
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->filters;
    l->filters = l->binary_filters;
    l->binary_filters = swap;

    #ifdef GPU
    swap = l->filters_gpu;
    l->filters_gpu = l->binary_filters_gpu;
    l->binary_filters_gpu = swap;
    #endif
}

void binarize_filters(float *filters, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(filters[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (filters[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}


int convolutional_out_depth(convolutional_layer l)
{
    int d = l.d;
    if (!l.pad) d -= l.size;
    else d -= 1;
    return d/l.stride + 1;
}


int convolutional_out_height(convolutional_layer l)
{
    int h = l.h;
    if (!l.pad) h -= l.size;
    else h -= 1;
    return h/l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    int w = l.w;
    if (!l.pad) w -= l.size;
    else w -= 1;
    return w/l.stride + 1;
}

size_t get_workspace_size(layer l){
	return (size_t)l.out_d*l.out_h*l.out_w*l.size*l.size*l.size*l.c*sizeof(float);
}

convolutional_layer make_convolutional_layer(int batch, int d, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch_normalize, int binary, int xnor)
{
    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

	l.d = d;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = pad;
    l.batch_normalize = batch_normalize;

    l.filters = calloc(c*n*size*size*size, sizeof(float));
    l.filter_updates = calloc(c*n*size*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*size*c));
    for(i = 0; i < c*n*size*size*size; ++i) l.filters[i] = scale*rand_uniform(-1, 1);
	int out_d = convolutional_out_depth(l);
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
	l.out_d = out_d;
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_d * l.out_h * l.out_w * l.out_c;
    l.inputs = l.d * l.w * l.h * l.c;

    l.output = calloc(l.batch * out_d * out_h * out_w * n, sizeof(float));
    l.delta  = calloc(l.batch * out_d * out_h * out_w * n, sizeof(float));

    if(binary){
        l.binary_filters = calloc(c*n*size*size*size, sizeof(float));
        l.cfilters = calloc(c*n*size*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_filters = calloc(c*n*size*size*size, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
    }
#ifdef GPU
    if(gpu_index >= 0){
        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.scales_gpu = cuda_make_array(l.scales, n);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "Convolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return l;
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_d = convolutional_out_depth(l);
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);//fill output with 0

    //if(l.xnor ){//?????
    //    binarize_filters(l.filters, l.n, l.c*l.size*l.size*l.size, l.binary_filters);
    //    swap_binary(&l);
    //    binarize_cpu(state.input, l.c*l.d*l.h*l.w*l.batch, l.binary_input);
    //    state.input = l.binary_input;
    //}

    int m = l.n;
	int k = l.size*l.size*l.size*l.c;
    int n = out_d*out_h*out_w;

    if (l.xnor && l.c%32 == 0 && AI2) {//???? not modified
        //forward_xnor_layer(l, state);
        printf("xnor\n");
    } else {

        float *a = l.filters;
        float *b = state.workspace;
        float *c = l.output;
        for(i = 0; i < l.batch; ++i){
            im2col_cpu(state.input, l.c, l.d, l.h, l.w,
                    l.size, l.stride, l.pad, b);
#ifdef DEBUG
			printf("------m=%d, n=%d, k=%d, l.c=%d-------\n",m,n,k,l.c);
			fflush(stdout);
#endif
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
#ifdef DEBUG
			printf("------gemm done-------\n");
			fflush(stdout);
#endif
            c += n*m;
            state.input += l.c*l.h*l.w*l.d;
        }
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, state);
    }
    add_bias(l.output, l.biases, l.batch, l.n, out_d*out_h*out_w);

    activate_array(l.output, m*n*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

void backward_convolutional_layer(convolutional_layer l, network_state state)
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.size*l.c;
    int k = convolutional_out_depth(l)*
		convolutional_out_height(l)*
        convolutional_out_width(l);

    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);

    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;
        float *b = state.workspace;
        float *c = l.filter_updates;

        float *im = state.input+i*l.c*l.d*l.h*l.w;

        im2col_cpu(im, l.c, l.d, l.h, l.w,
                l.size, l.stride, l.pad, b);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(state.delta){
            a = l.filters;
            b = l.delta + i*m*k;
            c = state.workspace;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            col2im_cpu(state.workspace, l.c, l.d, l.h, l.w, l.size, l.stride, l.pad, state.delta+i*l.c*l.d*l.h*l.w);
        }
    }
}

void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    axpy_cpu(size, -decay*batch, l.filters, 1, l.filter_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.filter_updates, 1, l.filters, 1);
    scal_cpu(size, momentum, l.filter_updates, 1);
}
