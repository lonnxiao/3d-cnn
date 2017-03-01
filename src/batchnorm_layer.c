/******************************************************************************
* filename: 
* file function: 
* last modified: 
*******************************************************************************/
#include "batchnorm_layer.h"
#include "blas.h"
#include <stdio.h>

layer make_batchnorm_layer(int batch, int w, int h, int d, int c)
{
    fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
    layer layer = {0};
    layer.type = BATCHNORM;
    layer.batch = batch;
    layer.d = layer.out_d = d;
    layer.h = layer.out_h = h;
    layer.w = layer.out_w = w;
    layer.c = layer.out_c = c;
    layer.output = calloc(d * h * w * c * batch, sizeof(float));
    layer.delta  = calloc(d * h * w * c * batch, sizeof(float));
    layer.inputs = w*h*d*c;
    layer.outputs = layer.inputs;

    layer.scales = calloc(c, sizeof(float));
    layer.scale_updates = calloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i){
        layer.scales[i] = 1;
    }

    layer.mean = calloc(c, sizeof(float));
    layer.variance = calloc(c, sizeof(float));

    layer.rolling_mean = calloc(c, sizeof(float));
    layer.rolling_variance = calloc(c, sizeof(float));
#ifdef GPU
    layer.output_gpu =  cuda_make_array(layer.output, d * h * w * c * batch);
    layer.delta_gpu =   cuda_make_array(layer.delta, d * h * w * c * batch);

    layer.scales_gpu = cuda_make_array(layer.scales, c);
    layer.scale_updates_gpu = cuda_make_array(layer.scale_updates, c);

    layer.mean_gpu = cuda_make_array(layer.mean, c);
    layer.variance_gpu = cuda_make_array(layer.variance, c);

    layer.rolling_mean_gpu = cuda_make_array(layer.mean, c);
    layer.rolling_variance_gpu = cuda_make_array(layer.variance, c);

    layer.mean_delta_gpu = cuda_make_array(layer.mean, c);
    layer.variance_delta_gpu = cuda_make_array(layer.variance, c);

    layer.x_gpu = cuda_make_array(layer.output, layer.batch*layer.outputs);
    layer.x_norm_gpu = cuda_make_array(layer.output, layer.batch*layer.outputs);
#endif
    return layer;
}

void forward_batchnorm_layer(layer l, network_state state)
{
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    if(l.type == CONNECTED){
        l.out_c = l.outputs;
        l.out_d = l.out_h = l.out_w = 1;
    }
    if(state.train){
        mean_cpu(l.output, l.batch, l.out_c, l.out_d*l.out_h*l.out_w, l.mean);   
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_d*l.out_h*l.out_w, l.variance);   
        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_d*l.out_h*l.out_w);   
    } else {
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_d*l.out_h*l.out_w);
    }
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_d*l.out_h*l.out_w);
}

void backward_batchnorm_layer(const layer layer, network_state state)
{
}

#ifdef GPU

void pull_batchnorm_layer(layer l)
{
    cuda_pull_array(l.scales_gpu, l.scales, l.c);
    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}
void push_batchnorm_layer(layer l)
{
    cuda_push_array(l.scales_gpu, l.scales, l.c);
    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}

void forward_batchnorm_layer_gpu(layer l, network_state state)
{
    if(l.type == BATCHNORM) copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    if(l.type == CONNECTED){
        l.out_c = l.outputs;
        l.out_d = l.out_h = l.out_w = 1;
    }
    if (state.train) {
        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_d*l.out_h*l.out_w, l.mean_gpu);
        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_d*l.out_h*l.out_w, l.variance_gpu);

        scal_ongpu(l.out_c, .95, l.rolling_mean_gpu, 1);
        axpy_ongpu(l.out_c, .05, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
        scal_ongpu(l.out_c, .95, l.rolling_variance_gpu, 1);
        axpy_ongpu(l.out_c, .05, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
        normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_d*l.out_h*l.out_w);
        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);
    } else {
        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_d*l.out_h*l.out_w);
    }

    scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_d*l.out_h*l.out_w);
}

void backward_batchnorm_layer_gpu(const layer l, network_state state)
{
    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h*l.out_d, l.scale_updates_gpu);

    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_d*l.out_h*l.out_w);

    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h*l.out_d, l.mean_delta_gpu);
    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h*l.out_d, l.variance_delta_gpu);
    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h*l.out_d, l.delta_gpu);
    if(l.type == BATCHNORM) copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);
}
#endif
