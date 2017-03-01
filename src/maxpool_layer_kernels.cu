#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "maxpool_layer.h"
#include "cuda.h"
}

__global__ void forward_maxpool_layer_kernel(int n, int in_d, int in_h, int in_w, int in_c, int stride, int size, int pad, float *input, float *output, int *indexes)
{
    int d = (in_d + 2*pad - size + 1)/stride + 1;
    int h = (in_h + 2*pad - size + 1)/stride + 1;
    int w = (in_w + 2*pad - size + 1)/stride + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int w_out = id % w;
    id /= w;
    int h_out = id % h;
    id /= h;
	int d_out = id % d;
	id /= d;
    int c_out = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad;
    int h_offset = -pad;
    int d_offset = -pad;

    int out_index = w_out + w*(h_out + h*(d_out + d*(c_out + c*b)));
    float max = -INFINITY;
    int max_i = -1;
    int k, l, m;
	for(k = 0; k < size; ++k){
		for(l = 0; l < size; ++l){
			for(m = 0; m < size; ++m){
				int cur_d = d_offset + d_out*stride + k;
				int cur_h = h_offset + h_out*stride + l;
				int cur_w = w_offset + w_out*stride + m;
				int index = cur_w + in_w*(cur_h + in_h*(cur_d + in_d*(c_out + b*in_c)));
				int valid = (cur_d >= 0 && cur_d < in_d &&
				        cur_h >= 0 && cur_h < in_h &&
						cur_w >= 0 && cur_w < in_w);
				float val = (valid != 0) ? input[index] : -INFINITY;
				max_i = (val > max) ? index : max_i;
				max   = (val > max) ? val   : max;
			}
		}
	}
    output[out_index] = max;
    indexes[out_index] = max_i;
}

__global__ void backward_maxpool_layer_kernel(int n, int in_d, int in_h, int in_w, int in_c, int stride, int size, int pad, float *delta, float *prev_delta, int *indexes)
{
    int d = (in_d + 2*pad - size + 1)/stride + 1;
    int h = (in_h + 2*pad - size + 1)/stride + 1;
    int w = (in_w + 2*pad - size + 1)/stride + 1;
    int c = in_c;
    int area = (size-1)/stride;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int index = id;
    int w_out = id % in_w;
    id /= in_w;
    int h_out = id % in_h;
    id /= in_h;
	int d_out = id % in_d;
    id /= in_d;
    int c_out = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = -pad;
    int h_offset = -pad;
    int d_offset = -pad;

    float tmp = 0;
    int k, l, m;
	for(k = -area; k < area+1; ++k)
		for(l = -area; l < area+1; ++l){
			for(m = -area; m < area+1; ++m){
				int out_w = (w_out-w_offset)/stride + m;
				int out_h = (h_out-h_offset)/stride + l;
				int out_d = (d_out-d_offset)/stride + k;
				int out_index = out_w + w*(out_h + h*(out_d + d*(c_out + c*b)));
				int valid = (out_w >= 0 && out_w < w &&
						 out_h >= 0 && out_h < h &&
						 out_d >= 0 && out_d < d);
				tmp += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
			}
		}
	}
    prev_delta[index] += tmp;
}

extern "C" void forward_maxpool_layer_gpu(maxpool_layer layer, network_state state)
{
	int d = layer.out_d;
    int h = layer.out_h;
    int w = layer.out_w;
    int c = layer.c;

    size_t n = d*h*w*c*layer.batch;

    forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.d, layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad, state.input, layer.output_gpu, layer.indexes_gpu);
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_maxpool_layer_gpu(maxpool_layer layer, network_state state)
{
    size_t n = layer.d*layer.h*layer.w*layer.c*layer.batch;

    backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.d, layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad, layer.delta_gpu, state.delta, layer.indexes_gpu);
    check_error(cudaPeekAtLastError());
}

