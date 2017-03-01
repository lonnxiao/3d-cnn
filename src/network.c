/******************************************************************************
* filename: 
* file function: 
* last modified: 
*******************************************************************************/
#include <stdio.h>
#include <time.h>
#include "network.h"
#include "image-3D.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

//#include "crop_layer.h"
#include "connected_layer.h"
//#include "gru_layer.h"
//#include "rnn_layer.h"
//#include "crnn_layer.h"
//#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
//#include "deconvolutional_layer.h"
//#include "detection_layer.h"
//#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
//#include "dropout_layer.h"
//#include "route_layer.h"
//#include "shortcut_layer.h"

typedef layer crop_layer;
typedef layer gru_layer;
typedef layer rnn_layer;
typedef layer crnn_layer;
typedef layer local_layer;
typedef layer deconvolutional_layer;
typedef layer detection_layer;
typedef layer normalization_layer;
typedef layer dropout_layer;
typedef layer route_layer;
typedef layer shortcut_layer;

#ifndef DEBUG
//#define DEBUG
#endif


int get_current_batch(network net)
{
    int batch_num = (*net.seen)/(net.batch*net.subdivisions);
    return batch_num;
}

void reset_momentum(network net)
{
    if (net.momentum == 0) return;
    net.learning_rate = 0;
    net.momentum = 0;
    net.decay = 0;
    #ifdef GPU
        if(gpu_index >= 0) update_network_gpu(net);
    #endif
}

float get_current_rate(network net)
{
    int batch_num = get_current_batch(net);
    int i;
    float rate;
    switch (net.policy) {
        case CONSTANT:
            return net.learning_rate;
        case STEP:
            return net.learning_rate * pow(net.scale, batch_num/net.step);
        case STEPS:
            rate = net.learning_rate;
            for(i = 0; i < net.num_steps; ++i){
                if(net.steps[i] > batch_num) return rate;
                rate *= net.scales[i];
                if(net.steps[i] > batch_num - 1) reset_momentum(net);
            }
            return rate;
        case EXP:
            return net.learning_rate * pow(net.gamma, batch_num);
        case POLY:
            if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
            return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
        case RANDOM:
            return net.learning_rate * pow(rand_uniform(0,1), net.power);
        case SIG:
            return net.learning_rate * (1./(1.+exp(net.gamma*(batch_num - net.step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net.learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.layers = calloc(net.n, sizeof(layer));
    net.seen = calloc(1, sizeof(int));
#ifdef GPU
    net.input_gpu = calloc(1, sizeof(float *));
    net.truth_gpu = calloc(1, sizeof(float *));
#endif
    return net;
}

/*************************************************************
* CNN forward train function
* 
***************************************************/
void forward_network(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta){
            scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
#ifdef DEBUG
		printf("-----i=%d-------",i);
		printf(" FORWARD LAYER TYPE: %s.\n", get_layer_string(l.type));
		fflush(stdout);
#endif
		if(l.type == CONVOLUTIONAL){
            forward_convolutional_layer(l, state);
        } else if(l.type == ACTIVE){
            forward_activation_layer(l, state);
        } else if(l.type == BATCHNORM){
            forward_batchnorm_layer(l, state);
        } else if(l.type == CONNECTED){
            forward_connected_layer(l, state);
        } else if(l.type == COST){
            forward_cost_layer(l, state);
        } else if(l.type == SOFTMAX){
            forward_softmax_layer(l, state);
        } else if(l.type == MAXPOOL){
            forward_maxpool_layer(l, state);
        } else if(l.type == AVGPOOL){
			forward_avgpool_layer(l, state);
		}else {
			printf("<FORWARD ERROR> NO SUCH LAYER TYPE: %s.\n", get_layer_string(l.type));
		}
        state.input = l.output;
    }
}

void update_network(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            update_convolutional_layer(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == CONNECTED){
            update_connected_layer(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

float *get_network_output(network net)
{
#ifdef GPU
	if (gpu_index >= 0) return get_network_output_gpu(net);
#endif 
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].output;
}

float get_network_cost(network net)
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].type == COST){
            sum += net.layers[i].cost[0];
            ++count;
        }
        if(net.layers[i].type == DETECTION){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    return sum/count;
}

void backward_network(network net, network_state state)
{
    int i;
    float *original_input = state.input;
    float *original_delta = state.delta;
    state.workspace = net.workspace;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output;
            state.delta = prev.delta;
        }
        layer l = net.layers[i];
#ifdef DEBUG
		printf("-----i=%d-------",i);
		printf(" BACKWARD LAYER TYPE: %s.\n", get_layer_string(l.type));
		fflush(stdout);
#endif
		if(l.type == CONVOLUTIONAL){
            backward_convolutional_layer(l, state);
        } else if(l.type == ACTIVE){
            backward_activation_layer(l, state);
        } else if(l.type == BATCHNORM){
            backward_batchnorm_layer(l, state);
        } else if(l.type == MAXPOOL){
            if(i != 0) backward_maxpool_layer(l, state);
        } else if(l.type == AVGPOOL){
            backward_avgpool_layer(l, state);
        } else if(l.type == SOFTMAX){
            if(i != 0) backward_softmax_layer(l, state);
        } else if(l.type == CONNECTED){
            backward_connected_layer(l, state);
        } else if(l.type == COST){
            backward_cost_layer(l, state);
        }else {
			printf("<BACKWARD ERROR> NO SUCH LAYER TYPE: %s.\n", get_layer_string(l.type));
		}
    }
}

float train_network_datum(network net, float *x, float *y)
{
#ifdef GPU
    if(gpu_index >= 0) return train_network_datum_gpu(net, x, y);
#endif
    *net.seen += net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    state.input = x;
    state.delta = 0;
    state.truth = y;
    state.train = 1;
	
    forward_network(net, state);
	
    backward_network(net, state);
    float error = get_network_cost(net);
    if(((*net.seen)/net.batch)%net.subdivisions == 0) update_network(net);
    return error;
}

float train_network(network net, data d)
{
    int batch = net.batch;	//train img number in one time
	int n = d.X.rows / batch;
#ifdef DEBUG
	printf("-----n = %d------\n",n);
	fflush(stdout);
#endif
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));
	
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, X, y);

        float err = train_network_datum(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
	
}

int get_network_output_size(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}

int get_network_input_size(network net)
{
    return net.layers[0].inputs;
}

float *network_predict(network net, float *input)
{
#ifdef GPU
    if(gpu_index >= 0)  return network_predict_gpu(net, input);
#endif
    network_state state;
    state.net = net;
    state.index = 0;
    state.input = input;//image.data
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network(net, state);
    float *out = get_network_output(net);
    return out;
}

matrix network_predict_data(network net, data test)
{
    int i,j,b;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void free_network(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        free_layer(net.layers[i]);
    }
    free(net.layers);
#ifdef GPU
    if(*net.input_gpu) cuda_free(*net.input_gpu);
    if(*net.truth_gpu) cuda_free(*net.truth_gpu);
    if(net.input_gpu) free(net.input_gpu);
    if(net.truth_gpu) free(net.truth_gpu);
#endif
}
