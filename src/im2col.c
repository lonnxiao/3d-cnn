/******************************************************************************
* filename: 
*******************************************************************************/
#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int depth, int height, int width, int channels,
                        int dep, int row, int col, int channel, int pad)
{
	dep -= pad;
    row -= pad;
    col -= pad;

    if (dep < 0 || row < 0 || col < 0 ||
        dep >= depth || row >= height || col >= width) return 0;
    return im[col + width*(row + height*(dep + depth*channel))];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int depth,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,d,h,w;
    int depth_col = (depth - ksize) / stride + 1;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    if (pad){
        depth_col = 1 + (depth-1) / stride;
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    int channels_col = channels * ksize * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int d_offset = (c / ksize / ksize) % ksize;
        int c_im = c / ksize / ksize / ksize;
		
		for (d = 0; d < depth_col; ++d){
			for (h = 0; h < height_col; ++h) {
				for (w = 0; w < width_col; ++w) {
					int im_dep = d_offset + d * stride;
					int im_row = h_offset + h * stride;
					int im_col = w_offset + w * stride;
					int col_index = ((c * depth + d) * height_col + h) * width_col + w;
					data_col[col_index] = im2col_get_pixel(data_im, depth, height, width, channels,
							im_dep, im_row, im_col, c_im, pad);
				}
			}
		}
    }
}
