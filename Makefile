GPU=0
CUDNN=0
DEBUG=0

ARCH= --gpu-architecture=compute_52 --gpu-code=compute_52

VPATH=./src/
EXEC=3DNet
OBJDIR=./obj/

CC=gcc
NVCC=nvcc
OPTS=-O2
LDFLAGS= -lm -pthread 
COMMON= 
CFLAGS=-Wall -Wfatal-errors 


ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

#cuda.o deconvolutional_layer.o crop_layer.o dropout_layer.o detection_layer.o captcha.o route_layer.o coco.o compare.o classifier.o 
#rnn_layer.o art.o gru_layer.o rnn.o rnn_vid.o crnn_layer.o demo.o tag.o cifar.o go.o local_layer.o swag.o shortcut_layer.o dice.o
#writing.o box.o nightmare.o normalization_layer.o image.o 
OBJ =gemm.o utils.o image-3D.o convolutional_layer.o list.o activations.o col2im.o im2col.o blas.o maxpool_layer.o softmax_layer.o data.o 
OBJ+=matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o avgpool_layer.o layer.o 
OBJ+=activation_layer.o batchnorm_layer.o read_3D_data.o cuda.o 3DNet.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o 
OBJ+=maxpool_layer_kernels.o softmax_layer_kernels.o network_kernels.o avgpool_layer_kernels.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj results $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

