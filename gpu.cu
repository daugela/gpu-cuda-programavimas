#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1920

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n) {
        out[index] = a[index] + b[index];
    }
}

int main()
{
    float *a, *b, *out;

    // Išskiriame atmintyje vietos trims kintamiesiems (masyvams)
    // "Float" slankaus kablelio skaičius N kartų
    cudaMallocManaged(&a, sizeof(float) * N);
    cudaMallocManaged(&b, sizeof(float) * N);
    cudaMallocManaged(&out, sizeof(float) * N);

    // Užpildome pradinius masyvus elementariais duomenimis
    // Dėl paprastumo - įvesties masyvai bus vienodi, o jų elementų reikšmės bus tiesiog jų indeksai
    // Svarbu atkreipti dėmesį, kas skaičiuojame nuo 0 - todėl maksimali reiškmė bus 1919 (N-1)
    for(int i = 0; i < N; i++){
        a[i] = (float)i;
        b[i] = (float)i;
    }

    // NVIDIA GeForce GTX 1070 turi 1920 cuda branduolius
    // Sukonfigūruojame CUDA "tinklelį" panaudojant bazinį N = 15x128
    int gpuGridSize = 15;
    int gpuBlockSize = 128;

    // Paleidžiame CUDA GPU dalies kodą (kernel)
    vector_add<<<gpuGridSize, gpuBlockSize>>>(out, a, b, N);

    // Pagrindinė C funkcija pagal nutylėjimą nelauktų GPU dalies kodo vykdymo pabaigos
    // Todėl turime aiškiai apibrėžti, kad prieš patikrinant rezultatus visos GPU gijos turi būti baigusios savo darbą
    cudaDeviceSynchronize();

    // Patikrinkime rezultatus
    for(int i = 0; i < N; i++){
        assert(out[i] == a[i] + b[i]);
    }

    // Sėkmingai suskaičiuota dviejų N ilgio vektorių suma [0.00, 2.00, 4.00, ... 3834.00, 3836.00, 3838.00]
    printf("Sėkmingai suskaičiuota dviejų N ilgio vektorių suma [%.2f, %.2f, %.2f, ... %.2f, %.2f, %.2f] \n", out[0], out[1], out[2], out[N-3], out[N-2], out[N-1]);

    // Atlaisviname kintamuosius ir baigiame programą
    cudaFree(a); 
    cudaFree(b); 
    cudaFree(out);

    return EXIT_SUCCESS;
}