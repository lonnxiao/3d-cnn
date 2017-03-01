/******************************************************************************
* filename: 
*******************************************************************************/
#include "data.h"
#include "utils.h"
#include "image-3D.h"
#include "read_3D_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DEBUG
//#define DEBUG
#endif
unsigned int data_seed;

void free_data(data d)
{
    if(d.indexes){
        free(d.indexes);
    }
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}

void *load_thread(void *ptr)
{
#ifdef GPU
    cudaError_t status = cudaSetDevice(gpu_index);
    check_error(status);
#endif
    //printf("Loading data: %d\n", rand_r(&data_seed));
    load_args a = *(struct load_args*)ptr;
	if (a.type == OLD_CLASSIFICATION_DATA){
		*a.dt = load_data(a.path, a.n, a.m, a.index, a.classes, a.w, a.h, a.d);
    }
    free(ptr);
    return 0;
}

pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

data load_data(char *path, int n, int m, int part, int k, int w, int h, int d)
{
	//int index;
	//if(m>0) index = rand_r(&data_seed)%(m-n+1);
	//else index = part;
#ifdef DEBUG
	printf("index=%d\n",part);
#endif
    data dt = {0};
    dt.shallow = 0;
	read_3D_data(&dt, path, n, part, k, w, h, d);
	return dt;
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
		//memcpy(void* dest, const void* src, size_t n);
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

