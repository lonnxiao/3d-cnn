/******************************************************************************
* filename: 3DNet.c
* file function: main function of the 3D-CNN project.
* history:
* 		07/29/2016: create the file.
*******************************************************************************/
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "network.h"
#include "utils.h"
#include "parser.h"
#include "cuda.h"

#ifndef DEBUG
//#define DEBUG
#endif

/******************************************************************************
* function: train_3DNet
* param: filename, cfg file path and name, like ./cfg/3DNet.cfg
* param: weightfile[optial], weight file path and name, like ./results/3DNet.weight
*******************************************************************************/
void train_3DNet(char *cfgfile, char *weightfile)
{
	FILE* file = fopen("train_log.txt","w");
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char *backup_directory = "./results";
    printf("%s\n", base);
    fprintf(file, "%s\n", base);
    network net = parse_network_cfg(cfgfile);
#ifdef DEBUG
	int i;
	for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
		fprintf(file, "<TEST NETWORK PARSE>LAYER TYPE: %s.\n", get_layer_string(l.type));
		printf("<TEST NETWORK PARSE>LAYER TYPE: %s.\n", get_layer_string(l.type));
	}
#endif
    if(weightfile){
        load_weights(&net, weightfile);
    }
    fprintf(file, "Learning Rate: %g, Momentum: %g, Decay: %g, max_batches: %d\n", net.learning_rate, net.momentum, net.decay, net.max_batches);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g, max_batches: %d\n", net.learning_rate, net.momentum, net.decay, net.max_batches);
    int imgs = 50;			//number of imgs LOADed one time
	char* path = "./3dNetworkDataset/shapenet10_train.bin";
	int N = 47892;
    clock_t time;
    pthread_t load_thread;
    data train;
    data buffer;

    load_args args = {0};
    args.w = net.w;			//3D objs' width
    args.h = net.h;			//3D objs' height
	args.d = net.d;			//3D objs' depth
    args.path = path;		//3D train dataset's path
    args.classes = 10;		//classes
	args.n = imgs;			//samples number of one batch
    args.m = N;				//training pic number 
	args.index = 0;
    //args.labels = labels;
    args.dt = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    load_thread = load_data_in_thread(args);		//use another thread to load data
#ifdef DEBUG
    printf("LOAD DATA DONE\n");
#endif // DEBUG
    int epoch = (*net.seen)/N;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        pthread_join(load_thread, 0);
        train = buffer;
#ifdef RANDOM_TRAIN_INDEX
		srand(time(0));
		args.index = rand()%(args.m-args.n+1);
#else
		args.index = (get_current_batch(net)*50)%(args.m-args.n+1);
#endif
        load_thread = load_data_in_thread(args);
        time=clock();
        float loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        fprintf(file, "%d, %d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", \
			   get_current_batch(net), args.index, (float)(*net.seen)/N, loss, avg_loss, \
			   get_current_rate(net), sec(clock()-time), *net.seen);
		printf("%d, %d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", \
			   get_current_batch(net), args.index, (float)(*net.seen)/N, loss, avg_loss, \
			   get_current_rate(net), sec(clock()-time), *net.seen);
        free_data(train);
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(*net.seen%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    pthread_join(load_thread, 0);
    free_data(buffer);
    free_network(net);
    free(base);
	fclose(file);
}

/******************************************************************************
* function: valid_3DNet 
* param: filename, cfg file path and name, like ./cfg/3DNet.cfg
* param: weightfile, weight file path and name, like ./results/3DNet.weight
*******************************************************************************/
void validate_3DNet(char *filename, char *weightfile)
{
    FILE* file = fopen("valid_log.txt", "w");
    int i = 0;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    char* path = "./3dNetworkDataset/shapenet10_test.bin";
    int m = 10896;

    clock_t time;
    int splits = 100;		//divided test data into 100 batches
    int num = m/splits;

    data val, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
	args.d = net.d;
    args.path = path;
    args.classes = 10;
    args.n = num;
    args.m = 0;				//train tag, 0 means valid
	args.index = 0;
    //args.labels = labels;
    args.dt = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
	float acc_avg = 0;
	int done_num = 0;
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
		args.n = num;
		int part = i*m/splits;
        if(i != splits){
            args.index = part;
            load_thread = load_data_in_thread(args);
        }
        fprintf(file,"Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
		matrix guess = network_predict_data(net, val);
		float acc;
#ifdef DEBUG
		int j,k;
		for(j=0;j<guess.rows;j++){
			for(k=0;k<guess.cols;k++){
				printf("%.2f ", guess.vals[j][k]);
			}
			printf("\n");
		}
#endif
		acc = matrix_topk_accuracy(val.y, guess, 1);
		acc_avg = (acc_avg*done_num + acc*num)/(done_num + num);
		done_num += num;
		
		printf("******%d: acc: %f, avg_acc: %f, %lf seconds, %d images\n", i, acc, acc_avg, sec(clock()-time), val.X.rows);
		fprintf(file, "******%d: acc: %f, avg_acc: %f, %lf seconds, %d images\n", i, acc, acc_avg, sec(clock()-time), val.X.rows);
		free_matrix(guess);
		free_data(val);
    }
	fclose(file);
}

/******************************************************************************
* main function
*******************************************************************************/
void main(int argc, char **argv)
{
    if(argc < 3){
        fprintf(stderr, "usage: %s [train/valid] [cfg] [weights (optional)]\n", argv[0]);
        return;
    }
	/**
	* gpu_index  = -1: no gpu
	* 			 >= 0: number of gpus
	* */
	gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }
#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
#endif

    char *cfg = argv[2];
    char *weights = (argc > 3) ? argv[3] : 0;
    if(0==strcmp(argv[1], "train")) train_3DNet(cfg, weights);
    else if(0==strcmp(argv[1], "valid")) validate_3DNet(cfg, weights);
}
