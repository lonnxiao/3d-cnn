/******************************************************************************
* filename: 
*******************************************************************************/
#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image-3D.h"
#include "layer.h"
#include "network.h"
#include "cuda.h"

typedef layer avgpool_layer;

avgpool_layer make_avgpool_layer(int batch, int w, int h, int d, int c);
void forward_avgpool_layer(const avgpool_layer l, network_state state);
void backward_avgpool_layer(const avgpool_layer l, network_state state);

#ifdef GPU
void forward_avgpool_layer_gpu(avgpool_layer l, network_state state);
void backward_avgpool_layer_gpu(avgpool_layer l, network_state state);
#endif

#endif

