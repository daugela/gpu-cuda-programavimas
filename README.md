# GPU programavimas su CUDA

Ši repozitorija yra pavyzdys, kaip elementarus ciklinis dviejų vektorių sumos algoritmas gali būti perrašytas pasitelkiant CUDA lygiagretaus skaičiavimo aplikacijų sąsają (API).  
Kodas nedemonstruoja greitaveikos rezultatų - jis skirtas tik suprasti esminius architektūrinius kodo pokyčius pradedant programuoti su CUDA ar optimizuojant jau turimus algoritmus.  
Išsamesnė informacija apie čia pateiktą turinį prieinama http://www.skaitmeninisturinys.lt/blog/gpu-cuda-programavimas  

## Kodas

cpu.c - pradinė paprasta C programa veikianti tik su CPU  
gpu.cu - cpu programa perrašyta naudojimui su GPU  

## Įranga

Reikalingas GCC kompiliatorius C kodui.
```
gcc --version
``` 
Reikalinga ne senesnė nei CUDA 6 versija (pavyzdyje naudoju Unified Memory)  
```
nvcc --version
```

Žinoma reikalingas NVIDIA GPU  
Aš naudojau savo turimą NVIDIA GTX 1070 su 1920 CUDA branduoliais  

## Susikompiliuojame be make failo

CPU dalies kodą:
```
gcc -o cpu cpu.c
```

GPU dalies kodą:
```
nvcc -o gpu gpu.cu
```

## Paleidžiame

CPU dalies kodą:
```
./cpu
```

GPU dalies kodą:
```
./gpu
```
