#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n) {
        out[index] = a[index] + b[index];
    }
}

int main()
{
    // Nustatome vektoriaus ilgį (narių skaičių)
    int N = 1<<20; // Didelis skaičius 1048576 arba 2^20

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

    // Sukonfigūruojame CUDA "tinklelį" panaudojant bazinį block_size 256
    // Ir paskaičiuojame kiek tokių blokų reikia (grid_size), kad siekti visą N ilgį
    int block_size = 256;
    int grid_size = N / block_size;

    // Paleidžiame CUDA GPU dalies kodą (kernel)
    vector_add<<<grid_size, block_size>>>(out, a, b, N);

    // Pagrindinė C funkcija pagal nutylėjimą nelauktų GPU dalies kodo vykdymo pabaigos
    // Todėl turime aiškiai apibrėžti, kad prieš patikrinant rezultatus visos GPU gijos turi būti baigusios savo darbą
    cudaDeviceSynchronize();

    // Patikrinkime rezultatus
    for(int i = 0; i < N; i++){
        assert(out[i] == a[i] + b[i]);
    }

    // Sėkmingai suskaičiuota dviejų N ilgio vektorių suma [0.00, 2.00, 4.00, ... 2097146.00, 2097148.00, 2097150.00]
    printf("Sėkmingai suskaičiuota dviejų N ilgio vektorių suma [%.2f, %.2f, %.2f, ... %.2f, %.2f, %.2f] \n", out[0], out[1], out[2], out[N-3], out[N-2], out[N-1]);

    // Atlaisviname kintamuosius ir baigiame programą
    cudaFree(a); 
    cudaFree(b); 
    cudaFree(out);

    return EXIT_SUCCESS;
}