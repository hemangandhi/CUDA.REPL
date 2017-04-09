#include <cuda_runtime.h>
#include <stdio.h>
__global__ void vec_add ( int * l, int * r, int n ){
int thid = threadIdx.x + blockDim.x * blockIdx.x;
if(thid >= n) return;
l[thid] += r[thid];
}
int main(){
int l[] = {1,2,3,4};
int * gpu_l;
cudaMalloc( &gpu_l, sizeof(int) * 4);
cudaMemcpy(gpu_l, l, sizeof(int) * 4, cudaMemcpyHostToDevice);
int r[] = {4, 3, 2, 1};
int * gpu_r;
cudaMalloc( &gpu_r, sizeof(int) * 4);
cudaMemcpy(gpu_r, r, sizeof(int) * 4, cudaMemcpyHostToDevice);
vec_add<<<2,2>>>( gpu_l,gpu_r,4);
printf("Last error: %s \n", cudaGetErrorString(cudaDeviceSynchronize()));
cudaMemcpy(l, gpu_l, sizeof(int) * 4, cudaMemcpyDeviceToHost);
cudaFree( gpu_l);
for(int i = 0; i < 4; i++)
printf("l[%d] : %d\n", i, l[i]);
cudaMemcpy(r, gpu_r, sizeof(int) * 4, cudaMemcpyDeviceToHost);
cudaFree( gpu_r);
for(int i = 0; i < 4; i++)
printf("r[%d] : %d\n", i, r[i]);
}