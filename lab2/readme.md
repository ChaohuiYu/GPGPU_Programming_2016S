
How to Compile it on CUDA 7.5:

nvcc main.cu counting.cu -std=c++11 -arch=sm_30 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__

