/******************************************************************************
* filename: 
*******************************************************************************/
#ifndef IM2COL_H
#define IM2COL_H

void im2col_cpu(float* data_im,
        int channels, int depth, int height, int width,
        int ksize, int stride, int pad, float* data_col);
#ifdef GPU
void im2col_ongpu(float *im,
				  int channels, int depth, int height, int width,
				  int ksize, int stride, int pad,float *data_col);

#endif
#endif
