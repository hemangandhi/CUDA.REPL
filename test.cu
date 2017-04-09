#include <cuda_runtime.h>
#include <stdio.h>
__global__ void origin ( int * n, int k ){
int thid = threadIdx.x + blockDim.x * blockIdx.x;
if (thid >= k) return;
n[thid] = thid;
}
int main(){
int n[] = {2,2,2,2};
int * gpu_n;
cudaMalloc( &gpu_n, sizeof(int) * 4);
cudaMemcpy(gpu_n, n, sizeof(int) * 4, cudaMemcpyHostToDevice);
origin<<<2,2>>>( gpu_n,4);
printf("Last error: %s \n", cudaGetErrorString(cudaDeviceSynchronize()));
cudaMemcpy(n, gpu_n, sizeof(int) * 4, cudaMemcpyDeviceToHost);
cudaFree( gpu_n);
for(int i = 0; i < 4; i++)
printf("n[%d] : %d\n", i, n[i]);
}