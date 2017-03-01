/******************************************************************************
* filename:
*******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "parser.h"
#include "activations.h"
#include "cost_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "batchnorm_layer.h"
#include "connected_layer.h"
#include "maxpool_layer.h"
#include "softmax_layer.h"
#include "avgpool_layer.h"
#include "list.h"
#include "option_list.h"
#include "utils.h"

//#define SAVE_WEIGHT_TXT

typedef struct{
    char *type;
    list *options;
}section;

int is_network(section *s);
int is_convolutional(section *s);
int is_activation(section *s);
int is_local(section *s);
int is_deconvolutional(section *s);
int is_connected(section *s);
int is_rnn(section *s);
int is_gru(section *s);
int is_crnn(section *s);
int is_maxpool(section *s);
int is_avgpool(section *s);
int is_dropout(section *s);
int is_softmax(section *s);
int is_normalization(section *s);
int is_batchnorm(section *s);
int is_crop(section *s);
int is_shortcut(section *s);
int is_cost(section *s);
int is_detection(section *s);
int is_route(section *s);
list *read_cfg(char *filename);

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i){
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}

typedef struct size_params{
    int batch;
    int inputs;
    int d;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
} size_params;

convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,d,h,w,c;
    batch=params.batch;
	d = params.d;
    h = params.h;
    w = params.w;
    c = params.c;
    if(!(d && h && w && c)) error("Layer before convolutional layer must output image.");

    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

	convolutional_layer layer = make_convolutional_layer(batch,d,h,w,c,n,size,stride,pad,activation, batch_normalize, binary, xnor);

    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    char *weights = option_find_str(options, "weights", 0);
    char *biases = option_find_str(options, "biases", 0);
    parse_data(weights, layer.filters, c*n*size*size*size);
    parse_data(biases, layer.biases, n);

    return layer;
}

connected_layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    connected_layer layer = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize);

    char *weights = option_find_str(options, "weights", 0);
    char *biases = option_find_str(options, "biases", 0);
    parse_data(biases, layer.biases, output);
    parse_data(weights, layer.weights, params.inputs*output);

    return layer;
}

softmax_layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups",1);
    softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);
    layer.temperature = option_find_float_quiet(options, "temperature", 1);
    return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    return layer;
}

maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);

	int batch,d,h,w,c;
	d = params.d;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(d && h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch,d,h,w,c,size,stride);

	return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,d,c;
    w = params.w;
    h = params.h;
    d = params.d;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.");

    avgpool_layer layer = make_avgpool_layer(batch,w,h,d,c);
    return layer;
}

layer parse_batchnorm(list *options, size_params params)
{
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.d, params.c);
    return l;
}

layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.out_d = params.d;
    l.out_h = params.h;
    l.out_w = params.w;
    l.out_c = params.c;
    l.d = params.d;
    l.h = params.h;
    l.w = params.w;
    l.c = params.c;

    return l;
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random")==0) return RANDOM;
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;

	net->d = option_find_int_quiet(options, "depth",0);
    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->h*net->w*2);//not sure
    net->min_crop = option_find_int_quiet(options, "min_crop",net->h*net->w);//not sure
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->d * net->c);
    if(!net->inputs && !(net->d && net->h && net->w && net->c)) error("No input parameters supplied");
    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS){
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
        net->power = option_find_float(options, "power", 1);
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

network parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections");
    network net = make_network(sections->size - 1);
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, &net);

    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
	params.d = net.d;

    params.inputs = net.inputs;
    params.batch = net.batch;
    params.time_steps = net.time_steps;

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    while(n){
        params.index = count;
        fprintf(stderr, "%d: ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = {0};

		if(is_convolutional(s)){
            l = parse_convolutional(options, params);
        }else if(is_activation(s)){
            l = parse_activation(options, params);
        }else if(is_connected(s)){
            l = parse_connected(options, params);
        }else if(is_cost(s)){
            l = parse_cost(options, params);
        }else if(is_softmax(s)){
            l = parse_softmax(options, params);
        }else if(is_batchnorm(s)){
            l = parse_batchnorm(options, params);
        }else if(is_maxpool(s)){
            l = parse_maxpool(options, params);
        }else if(is_avgpool(s)){
            l = parse_avgpool(options, params);
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }

        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        option_unused(options);
        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
			params.d = l.out_d;
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    net.outputs = get_network_output_size(net);
    net.output = get_network_output(net);
    if(workspace_size){
		//printf("%ld\n", workspace_size);
#ifdef GPU
        if(gpu_index >= 0){
            net.workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }else {
            net.workspace = calloc(1, workspace_size);
        }
#else
        net.workspace = calloc(1, workspace_size);
#endif
    }
    return net;
}

LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[deconv]")==0
            || strcmp(type, "[deconvolutional]")==0) return DECONVOLUTIONAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[route]")==0) return ROUTE;
    return BLANK;
}

int is_shortcut(section *s)
{
    return (strcmp(s->type, "[shortcut]")==0);
}
int is_crop(section *s)
{
    return (strcmp(s->type, "[crop]")==0);
}
int is_cost(section *s)
{
    return (strcmp(s->type, "[cost]")==0);
}
int is_detection(section *s)
{
    return (strcmp(s->type, "[detection]")==0);
}
int is_local(section *s)
{
    return (strcmp(s->type, "[local]")==0);
}
int is_deconvolutional(section *s)
{
    return (strcmp(s->type, "[deconv]")==0
            || strcmp(s->type, "[deconvolutional]")==0);
}
int is_convolutional(section *s)
{
    return (strcmp(s->type, "[conv]")==0
            || strcmp(s->type, "[convolutional]")==0);
}
int is_activation(section *s)
{
    return (strcmp(s->type, "[activation]")==0);
}
int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}
int is_crnn(section *s)
{
    return (strcmp(s->type, "[crnn]")==0);
}
int is_gru(section *s)
{
    return (strcmp(s->type, "[gru]")==0);
}
int is_rnn(section *s)
{
    return (strcmp(s->type, "[rnn]")==0);
}
int is_connected(section *s)
{
    return (strcmp(s->type, "[conn]")==0
            || strcmp(s->type, "[connected]")==0);
}
int is_maxpool(section *s)
{
    return (strcmp(s->type, "[max]")==0
            || strcmp(s->type, "[maxpool]")==0);
}
int is_avgpool(section *s)
{
    return (strcmp(s->type, "[avg]")==0
            || strcmp(s->type, "[avgpool]")==0);
}
int is_dropout(section *s)
{
    return (strcmp(s->type, "[dropout]")==0);
}

int is_normalization(section *s)
{
    return (strcmp(s->type, "[lrn]")==0
            || strcmp(s->type, "[normalization]")==0);
}

int is_batchnorm(section *s)
{
    return (strcmp(s->type, "[batchnorm]")==0);
}

int is_softmax(section *s)
{
    return (strcmp(s->type, "[soft]")==0
            || strcmp(s->type, "[softmax]")==0);
}
int is_route(section *s)
{
    return (strcmp(s->type, "[route]")==0);
}

list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *sections = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = malloc(sizeof(section));
                list_insert(sections, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return sections;
}

void save_weights_double(network net, char *filename)
{
    fprintf(stderr, "Saving doubled weights to %s\n", filename);
    FILE *fp = fopen(filename, "w");
    if(!fp) file_error(filename);

    fwrite(&net.learning_rate, sizeof(float), 1, fp);
    fwrite(&net.momentum, sizeof(float), 1, fp);
    fwrite(&net.decay, sizeof(float), 1, fp);
    fwrite(net.seen, sizeof(int), 1, fp);

    int i,j,k;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_convolutional_layer(l);
            }
#endif
            float zero = 0;
            fwrite(l.biases, sizeof(float), l.n, fp);
            fwrite(l.biases, sizeof(float), l.n, fp);

            for (j = 0; j < l.n; ++j){
                int index = j*l.c*l.size*l.size*l.size;
                fwrite(l.filters+index, sizeof(float), l.c*l.size*l.size*l.size, fp);
                for (k = 0; k < l.c*l.size*l.size*l.size; ++k) fwrite(&zero, sizeof(float), 1, fp);
            }
            for (j = 0; j < l.n; ++j){
                int index = j*l.c*l.size*l.size*l.size;
                for (k = 0; k < l.c*l.size*l.size*l.size; ++k) fwrite(&zero, sizeof(float), 1, fp);
                fwrite(l.filters+index, sizeof(float), l.c*l.size*l.size*l.size, fp);
            }
        }
    }
    fclose(fp);
}

void save_convolutional_weights_binary(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    binarize_filters(l.filters, l.n, l.c*l.size*l.size*l.size, l.binary_filters);
    int size = l.c*l.size*l.size*l.size;
    int i, j, k;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    for(i = 0; i < l.n; ++i){
        float mean = l.binary_filters[i*size];
        if(mean < 0) mean = -mean;
        fwrite(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                if (l.binary_filters[index + k] > 0) c = (c | 1<<k);
            }
            fwrite(&c, sizeof(char), 1, fp);
        }
    }
}

void save_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int num = l.n*l.c*l.size*l.size*l.size;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.filters, sizeof(float), num, fp);
}

void save_batchnorm_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_batchnorm_layer(l);
    }
#endif
    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}


void save_weights_upto(network net, char *filename, int cutoff)
{
#ifdef GPU
    cuda_set_device(net.gpu_index);
#endif
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "w");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 1;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net.seen, sizeof(int), 1, fp);

    int i;
    for(i = 0; i < net.n && i < cutoff; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            save_convolutional_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
			/*
        } if(l.type == RNN){
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        } if(l.type == GRU){
            save_connected_weights(*(l.input_z_layer), fp);
            save_connected_weights(*(l.input_r_layer), fp);
            save_connected_weights(*(l.input_h_layer), fp);
            save_connected_weights(*(l.state_z_layer), fp);
            save_connected_weights(*(l.state_r_layer), fp);
            save_connected_weights(*(l.state_h_layer), fp);
        } if(l.type == CRNN){
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
        } if(l.type == LOCAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_local_layer(l);
            }
#endif
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.size*l.c*l.n*locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.filters, sizeof(float), size, fp);
			*/
        }
    }
    fclose(fp);

#ifdef SAVE_WEIGHT_TXT
	FILE *fp_log = fopen("weight.txt", "w");
    fprintf(fp_log, "major = %d \n", 0);
    fprintf(fp_log, "minor = %d \n", 1);
    fprintf(fp_log, "revision = %d \n", 0);
    fprintf(fp_log, "net.seen = %d \n", *(net.seen));
	fflush(stdout);

	long int j;
	int i_log;
    for(i_log = 0; i_log < net.n && i_log < cutoff; ++i_log){
        layer l_log = net.layers[i_log];
        if(l_log.type == CONVOLUTIONAL){
			int num_log = l_log.n*l_log.c*l_log.size*l_log.size*l_log.size;
			fprintf(fp_log, "l.type = CONVOLUTIONAL \n");
			fprintf(fp_log, "l.biases : \n");
			for(j=0; j!=l_log.n; ++j)
				fprintf(fp_log, "%f ", *(l_log.biases+j));
			fprintf(fp_log, "\n");
			if (l_log.batch_normalize){
				fprintf(fp_log, "l.scales : \n");
				for(j=0; j!=l_log.n; ++j)
					fprintf(fp_log, "%f ", *(l_log.scales+j));
				fprintf(fp_log, "\n");

				fprintf(fp_log, "l.rolling_mean : \n");
				for(j=0; j!=l_log.n; ++j)
					fprintf(fp_log, "%f ", *(l_log.rolling_mean+j));
				fprintf(fp_log, "\n");

				fprintf(fp_log, "l.rolling_variance : \n");
				for(j=0; j!=l_log.n; ++j)
					fprintf(fp_log, "%f ", *(l_log.rolling_variance+j));
				fprintf(fp_log, "\n");
			}
			fprintf(fp_log, "l.filters : \n");
			for(j=0; j!=num_log; ++j)
				fprintf(fp_log, "%f ", *(l_log.filters+j));
			fprintf(fp_log, "\n");
        } if(l_log.type == CONNECTED){
			fprintf(fp_log, "l.type = CONNECTED \n");

			fprintf(fp_log, "l.biases : \n");
			for(j=0; j!=l_log.outputs; ++j)
				fprintf(fp_log, "%f ", *(l_log.biases+j));
			fprintf(fp_log, "\n");

			fprintf(fp_log, "l.weights : \n");
			for(j=0; j!=l_log.outputs*l_log.inputs; ++j)
				fprintf(fp_log, "%f ", *(l_log.weights+j));
			fprintf(fp_log, "\n");

			if (l_log.batch_normalize){
				fprintf(fp_log, "l.scales : \n");
				for(j=0; j!=l_log.outputs; ++j)
					fprintf(fp_log, "%f ", *(l_log.scales+j));
				fprintf(fp_log, "\n");

				fprintf(fp_log, "l.rolling_mean : \n");
				for(j=0; j!=l_log.outputs; ++j)
					fprintf(fp_log, "%f ", *(l_log.rolling_mean+j));
				fprintf(fp_log, "\n");

				fprintf(fp_log, "l.rolling_variance : \n");
				for(j=0; j!=l_log.outputs; ++j)
					fprintf(fp_log, "%f ", *(l_log.rolling_variance+j));
				fprintf(fp_log, "\n");
			}
        } if(l_log.type == BATCHNORM){
			fprintf(fp_log, "l.type = BATCHNORM \n");

			fprintf(fp_log, "l.scales : \n");
			for(j=0; j!=l_log.c; ++j)
				fprintf(fp_log, "%f ", *(l_log.scales+j));
			fprintf(fp_log, "\n");

			fprintf(fp_log, "l.rolling_mean : \n");
			for(j=0; j!=l_log.c; ++j)
				fprintf(fp_log, "%f ", *(l_log.rolling_mean+j));
			fprintf(fp_log, "\n");

			fprintf(fp_log, "l.rolling_variance : \n");
			for(j=0; j!=l_log.c; ++j)
				fprintf(fp_log, "%f ", *(l_log.rolling_variance+j));
			fprintf(fp_log, "\n");
		}
    }
    fclose(fp_log);
#endif
}
void save_weights(network net, char *filename)
{
    save_weights_upto(net, filename, net.n);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer(l);
    }
#endif
}

void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_batchnorm_layer(l);
    }
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    int size = l.c*l.size*l.size*l.size;
    int i, j, k;
    for(i = 0; i < l.n; ++i){
        float mean = 0;
        fread(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            fread(&c, sizeof(char), 1, fp);
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                l.filters[index + k] = (c & 1<<k) ? mean : -mean;
            }
        }
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l.n*l.c*l.size*l.size*l.size;
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fread(l.filters, sizeof(float), num, fp);
    if (l.flipped) {
        transpose_matrix(l.filters, l.c*l.size*l.size*l.size, l.n);
    }
    //if (l.binary) binarize_filters(l.filters, l.n, l.c*l.size*l.size, l.filters);
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}


void load_weights_upto(network *net, char *filename, int cutoff)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
#endif
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    fread(net->seen, sizeof(int), 1, fp);
    int transpose = (major > 1000) || (minor > 1000);
#ifdef SAVE_WEIGHT_TXT
    FILE *fp_log = fopen("weight_rewrite.txt", "w");
	long int j;
	fprintf(fp_log, "major = %d \n", major);
    fprintf(fp_log, "minor = %d \n", major);
    fprintf(fp_log, "revision = %d \n", major);
#endif
    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL){
            load_convolutional_weights(l, fp);
#ifdef SAVE_WEIGHT_TXT
			int num = l.n*l.c*l.size*l.size*l.size;
			fprintf(fp_log, "l.type = CONVOLUTIONAL \n");
			fprintf(fp_log, "l.biases : \n");
			for(j=0; j!=l.n; ++j)
				fprintf(fp_log, "%f ", *(l.biases+j));
			fprintf(fp_log, "\n");
			if (l.batch_normalize){
				fprintf(fp_log, "l.scales : \n");
				for(j=0; j!=l.n; ++j)
					fprintf(fp_log, "%f ", *(l.scales+j));
				fprintf(fp_log, "\n");

				fprintf(fp_log, "l.rolling_mean : \n");
				for(j=0; j!=l.n; ++j)
					fprintf(fp_log, "%f ", *(l.rolling_mean+j));
				fprintf(fp_log, "\n");

				fprintf(fp_log, "l.rolling_variance : \n");
				for(j=0; j!=l.n; ++j)
					fprintf(fp_log, "%f ", *(l.rolling_variance+j));
				fprintf(fp_log, "\n");
				}
			fprintf(fp_log, "l.filters : \n");
			for(j=0; j!=num; ++j)
				fprintf(fp_log, "%f ", *(l.filters+j));
			fprintf(fp_log, "\n");
#endif
        }
		/*
        if(l.type == DECONVOLUTIONAL){
            int num = l.n*l.c*l.size*l.size*l.size;
            fread(l.biases, sizeof(float), l.n, fp);
            fread(l.filters, sizeof(float), num, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_deconvolutional_layer(l);
            }
#endif
        }*/
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
#ifdef SAVE_WEIGHT_TXT
			fprintf(fp_log, "l.type = CONNECTED \n");

			fprintf(fp_log, "l.biases : \n");
			for(j=0; j!=l.outputs; ++j)
				fprintf(fp_log, "%f ", *(l.biases+j));
			fprintf(fp_log, "\n");

			fprintf(fp_log, "l.weights : \n");
			for(j=0; j!=l.outputs*l.inputs; ++j)
				fprintf(fp_log, "%f ", *(l.weights+j));
			fprintf(fp_log, "\n");

			if (l.batch_normalize){
				fprintf(fp_log, "l.scales : \n");
				for(j=0; j!=l.outputs; ++j)
					fprintf(fp_log, "%f ", *(l.scales+j));
				fprintf(fp_log, "\n");

				fprintf(fp_log, "l.rolling_mean : \n");
				for(j=0; j!=l.outputs; ++j)
					fprintf(fp_log, "%f ", *(l.rolling_mean+j));
				fprintf(fp_log, "\n");

				fprintf(fp_log, "l.rolling_variance : \n");
				for(j=0; j!=l.outputs; ++j)
					fprintf(fp_log, "%f ", *(l.rolling_variance+j));
				fprintf(fp_log, "\n");
			}
#endif
        }
        if(l.type == BATCHNORM){
            load_batchnorm_weights(l, fp);
#ifdef SAVE_WEIGHT_TXT
			fprintf(fp_log, "l.type = BATCHNORM \n");

			fprintf(fp_log, "l.scales : \n");
			for(j=0; j!=l.c; ++j)
				fprintf(fp_log, "%f ", *(l.scales+j));
			fprintf(fp_log, "\n");

			fprintf(fp_log, "l.rolling_mean : \n");
			for(j=0; j!=l.c; ++j)
				fprintf(fp_log, "%f ", *(l.rolling_mean+j));
			fprintf(fp_log, "\n");

			fprintf(fp_log, "l.rolling_variance : \n");
			for(j=0; j!=l.c; ++j)
				fprintf(fp_log, "%f ", *(l.rolling_variance+j));
			fprintf(fp_log, "\n");

#endif
        }
		/*
        if(l.type == CRNN){
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == RNN){
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if(l.type == GRU){
            load_connected_weights(*(l.input_z_layer), fp, transpose);
            load_connected_weights(*(l.input_r_layer), fp, transpose);
            load_connected_weights(*(l.input_h_layer), fp, transpose);
            load_connected_weights(*(l.state_z_layer), fp, transpose);
            load_connected_weights(*(l.state_r_layer), fp, transpose);
            load_connected_weights(*(l.state_h_layer), fp, transpose);
        }
        if(l.type == LOCAL){
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.filters, sizeof(float), size, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_local_layer(l);
            }
#endif
        }
		*/
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
#ifdef SAVE_WEIGHT_TXT
	fclose(fp_log);
#endif
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, net->n);
}

