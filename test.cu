#include "cudaRuntime.h"
__global__ void default (  int * wat  ){
int thid = threadIdx.x + blockDim.x * blockIdx.x;

wat[thid] = thid;

}
int main(){
int wat[] = {};
int * gpu_wat;
cudaMalloc( &gpu_wat, sizeof(int) * 1);
cudaMemcpy(gpu_wat, wat, sizeof(int) * 1, cudaMemcpyHostToDevice);
default<<<,>>>( gpu_,);
printf("Last error: %s \n", cudaGetErrorString(cudaDeviceSynchronize()));
cudaMemcpy(wat, gpu_wat, sizeof(int) * 1, cudaMemcpyDeviceToHost);
cudaFree( gpu_wat);
for(int i = 0; i < 1; i++)
printf("wat[%d] : \n", i, wat[i]);
}