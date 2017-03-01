/******************************************************************************
* filename: 
* file function: 
* last modified: 
*******************************************************************************/
#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

//#include "cuda.h"
#include "image-3D.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer convolutional_layer;
#ifdef GPU
void forward_convolutional_layer_gpu(convolutional_layer layer, network_state state);
void backward_convolutional_layer_gpu(convolutional_layer layer, network_state state);
void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay);

void push_convolutional_layer(convolutional_layer layer);
void pull_convolutional_layer(convolutional_layer layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
//void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l);
#endif
#endif
convolutional_layer make_convolutional_layer(int batch, int d, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch_normalization, int binary, int xnor);

void forward_convolutional_layer(const convolutional_layer layer, network_state state);
void update_convolutional_layer(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay);
void binarize_filters(float *filters, int n, int size, float *binary);
void swap_binary(convolutional_layer *l);

void backward_convolutional_layer(convolutional_layer layer, network_state state);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);

#endif

