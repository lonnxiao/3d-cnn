/******************************************************************************
* filename: data.h
*******************************************************************************/
#ifndef DATA_H
#define DATA_H
#include <pthread.h>

#include "matrix.h"
#include "list.h"
#include "image-3D.h"

extern unsigned int data_seed;

typedef struct{
    int w, h, d;	
    matrix X;
    matrix y;
    int *indexes;
    int shallow;
    int *num_boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA
} data_type;

typedef struct load_args{
    char **paths;
    char *path;
    int n;
    int m;
	int index;
	int d;
    int h;
    int w;
	int out_d;
    int out_w;
    int out_h;
	int nd;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    float jitter;
    data *dt;
    image *im;
    image *resized;
    data_type type;
} load_args;

void free_data(data d);

pthread_t load_data_in_thread(load_args args);

data load_data(char *paths, int n, int m, int part, int k, int w, int h, int d);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void translate_data_rows(data d, float s);

#endif
