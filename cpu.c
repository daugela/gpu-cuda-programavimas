#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i ++){
        out[i] = a[i] + b[i];
    }
}

int main()
{
    // Nustatome vektoriaus ilgį (narių skaičių)
    int N = 1<<20; // Didelis skaičius 1048576 arba 2^20

    float *a, *b, *out;

    // Išskiriame atmintyje vietos trims kintamiesiems (masyvams)
    // "Float" slankaus kablelio skaičius N kartų 
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Užpildome pradinius masyvus elementariais duomenimis
    // Dėl paprastumo - įvesties masyvai bus vienodi, o jų elementų reikšmės bus tiesiog jų indeksai
    // Svarbu atkreipti dėmesį, kas skaičiuojame nuo 0 - todėl maksimali reiškmė bus 1919 (N-1)
    for(int i = 0; i < N; i++){
        a[i] = (float)i;
        b[i] = (float)i;
    }

    // Paleidžiame Kernel funkciją
    vector_add(out, a, b, N);

    // Patikrinkime rezultatus
    for(int i = 0; i < N; i++){
        assert(out[i] == a[i] + b[i]);
    }

    // Sėkmingai suskaičiuota dviejų N ilgio vektorių suma [0.00, 2.00, 4.00, ... 2097146.00, 2097148.00, 2097150.00]
    printf("Sėkmingai suskaičiuota dviejų N ilgio vektorių suma [%.2f, %.2f, %.2f, ... %.2f, %.2f, %.2f] \n", out[0], out[1], out[2], out[N-3], out[N-2], out[N-1]);

    // Atlaisviname kintamuosius ir baigiame programą
    free(a); 
    free(b); 
    free(out);

    return EXIT_SUCCESS;
}