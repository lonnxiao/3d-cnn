/******************************************************************************
* filename: read_3D_data.c
* file function: read the 3D data from path(filename)
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "read_3D_data.h"
#include "data.h"

/****************************************************************
* function: read <num> 3d objects data from <start_pos>
* INPUT: 
****************************************************************/
void read_3D_data(data *dt, char* filename, int num, int start_pos, int classes, int w, int h, int d)
{
    //char filename[] = "./shapenet10_train.bin";
    int num_instance;//number of 3D obj
    int shape_all;//3D obj data, = 32*32*32
	int size = w*h*d;
    FILE *fp = fopen(filename,"rb");

    if (fp == 0)
    {
        printf("File open failed.\n");
        return;
    }

    fread(&num_instance,sizeof(int),1,fp);
    fread(&shape_all,sizeof(int),1,fp);
    
	if(size != shape_all)
	{
		printf("Wrong w,h,d size!\n");
		return;
	}
	//printf("num_instance:%d shape_all:%d\n",num_instance,shape_all);
    
	matrix X = make_matrix(num, shape_all);
	matrix y = make_matrix(num, classes);
	
    //float *data = (float *)calloc(num_instance*shape_all,sizeof(float));
    //int *label = (int *)calloc(num_instance,sizeof(int));

    int i,index;
	for (i=0;i<start_pos;++i){
		fread(X.vals[0], sizeof(float), shape_all, fp);
		fread(&index, sizeof(int), 1, fp);
	}
    for (i=0;i<num;++i){
        fread(X.vals[i],sizeof(float),shape_all,fp);
		
		fread(&index,sizeof(int),1,fp);
		//printf("The label is %d\n",index);
		memset(y.vals[i], 0, classes*sizeof(float));
		y.vals[i][index] = 1;
    }
	
	dt->X = X;
	dt->y = y;
    fclose(fp);
}
