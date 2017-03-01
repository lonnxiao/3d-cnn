#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col.h"
#include "cuda.h"
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int depth, const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int depth_col, const int height_col, const int width_col,
        float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
		int d_index = h_index / height_col;
		int d_out = d_index % depth_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int d_in = d_out * stride - pad;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += ((channel_out * depth_col + d_out) * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += ((channel_in * depth + d_in)* height + h_in) * width + w_in;
		for(int k = 0; k < ksize; ++k) {
			for (int i = 0; i < ksize; ++i) {
				for (int j = 0; j < ksize; ++j) {
					int d = d_in + k;
					int h = h_in + i;
					int w = w_in + j;

					*data_col_ptr = (d >=0 && h >= 0 && w >= 0 && h < height && w < width && d < depth) ?
						data_im_ptr[k * height * width + i * width + j] : 0;

					//*data_col_ptr = data_im_ptr[ii * width + jj];

					data_col_ptr += depth_col * height_col * width_col;
				}
			}
		}
    }
}

void im2col_ongpu(float *im,
         int channels, int depth, int height, int width,
         int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int depth_col = (depth + 2 * pad - ksize) / stride + 1;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * depth_col * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, depth, height, width, ksize, pad,
                stride, depth_col, height_col,
                width_col, data_col);
}
