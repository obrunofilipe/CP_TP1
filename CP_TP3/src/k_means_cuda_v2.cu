#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <chrono>
#include <time.h>

using namespace std;

int N;
int K;

typedef struct ponto{
    float x;
    float y;
    int cluster;
} Ponto;

typedef struct centroide{
    float x;
    float y;
    float soma_x;
    float soma_y;
    int total_pontos;
} Centroide;




__global__
void kMeansKernel (Ponto *d_points, Centroide *d_centroids) { // código executado no GPU
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    int N = 10000000;
    int K = 4;

    if (id < N){
        Ponto point = d_points[id];
        //d_points[id].cluster = 1;
        
        float d = 10000.0f;
        int cluster = -1;

        for(int j = 0; j < 4; j++){
            
            float tmp = (point.x - d_centroids[j].x) * (point.x- d_centroids[j].x) + 
                        (point.y - d_centroids[j].y) * (point.y - d_centroids[j].y);
            if (tmp < d){
                d = tmp;
                cluster = j;
            }
        }

        int cl = point.cluster; // obtençao do cluster em que estava o ponto

        if(cl != cluster && cluster != -1){
            d_points[id].cluster = cluster;
        }
    }

    if(id<K){
        d_centroids[id].soma_x = 0;
        d_centroids[id].soma_y = 0;
        d_centroids[id].total_pontos = 0;
    }
    
    int cluster = d_points[id].cluster;
    atomicAdd(&d_centroids[cluster].soma_x,d_points[id].x);
    atomicAdd(&d_centroids[cluster].soma_y,d_points[id].y);
    atomicAdd(&d_centroids[cluster].total_pontos,1);
    
    
    
}

__global__
void newCentroidsKernel (Ponto* d_points, Centroide* d_centroids){

    int id = threadIdx.x;
    
    d_centroids[id].x = d_centroids[id].soma_x/d_centroids[id].total_pontos;
    d_centroids[id].y = d_centroids[id].soma_y/d_centroids[id].total_pontos;


}



void init_pontos(Ponto** p, Centroide** c){
     
    *p = (Ponto *)malloc(sizeof( Ponto ) * N);
    *c = (Centroide *)malloc(sizeof( struct centroide ) * K);
    
    srand(10); //inicialização dos pontos
    for (int i = 0; i < N; i++){
        (*p)[i].x = (float) rand() / RAND_MAX;
        (*p)[i].y = (float) rand() / RAND_MAX;
        (*p)[i].cluster = -1;
    }
    
    for(int i  = 0; i < K ; i++){ // atribuição dos primeiros valores do centroid dos clusters
        (*c)[i].x = (*p)[i].x;
        (*c)[i].y = (*p)[i].y;
        (*c)[i].soma_x = (*p)[i].x;
        (*c)[i].soma_y = (*p)[i].y;
        (*c)[i].total_pontos = 1;
    }

    
}


void new_centroids(Ponto* points, Centroide* centroides){// para cada centroide, vai ser calculado o novo valor das suas coordenadas, com base na média geométrica dos pontos que estão no cluster correspondente
    
    //inicializar os centroides a 0 novamente
    for (int i = 0; i < K; i++){
        centroides[i].soma_x = 0;
        centroides[i].soma_y = 0;
        centroides[i].total_pontos = 0;
    }
    
    //calcular a soma de pontos e nº de pontos para cada centroide
    for(int i = 0; i < N; i++){
        int cluster = points[i].cluster;
        centroides[cluster].soma_x += points[i].x;
        centroides[cluster].soma_y += points[i].y;
        centroides[cluster].total_pontos++;
    }
    
    //calcular os novos centroides
    for(int i = 0; i < K; i++){
        centroides[i].x = centroides[i].soma_x / centroides[i].total_pontos;
        centroides[i].y = centroides[i].soma_y / centroides[i].total_pontos;
    }
}


void printPoints(Ponto* points, Centroide* centroids, int n){
    for(int i = 0; i < n; i++){
        printf("Ponto %d :: (%f,%f) -> %d\n", i, points[i].x, points[i].y, points[i].cluster);
    }
    for(int i = 0; i < 4; i++){
        printf("Centroide %d :: (%f,%f) -> %d\n", i, centroids[i].soma_x, centroids[i].soma_y, centroids[i].total_pontos);
    }
}


int k_means(Ponto* points, Centroide* centroids){

    //int changed_some_point = 1;
    int n_iter = 0;


    // pointer to the device memory
    Ponto *d_points;
    Centroide *d_centroids;


    // allocate memory on the device
    cudaMalloc ( &d_points, N*sizeof(Ponto));
	cudaMalloc ( &d_centroids, K*sizeof(Centroide));


    //copy points to device
    cudaMemcpy(d_points, points, N*sizeof(Ponto), cudaMemcpyHostToDevice);
    //copy centroids to device
    cudaMemcpy(d_centroids, centroids, K*sizeof(Centroide), cudaMemcpyHostToDevice);


    while(/*changed_some_point &&*/ n_iter < 20){// enquanto há pontos a mudarem de cluster
        //changed_some_point = 0;

        // launch the kernel
	    //startKernelTime ();
	    kMeansKernel <<< 20000, 500 >>> (d_points, d_centroids); // atribuit os pontos a centroides

        newCentroidsKernel <<< 1, 4 >>> (d_points, d_centroids); //calcular os novos centroides

        //printPoints(points, centroids, 20);
        //new_centroids(points, centroids);
        n_iter++;
    }

    cudaMemcpy(centroids, d_centroids, K*sizeof(Centroide), cudaMemcpyDeviceToHost);

    // free the device memory
    cudaFree(d_points);
    cudaFree(d_centroids);


    return n_iter;
}




int main(int argc, char** argv){
    Ponto * pontos;
    Centroide * centroides;
    
    N = atoi(argv[1]);
    K = atoi(argv[2]);
    
    init_pontos(&pontos, &centroides);

    //printPoints(pontos, centroides, 20);

    clock_t start, end;
    double elapsed;

    start = clock();

    int n_iter = k_means(pontos,centroides);

    end = clock();
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %lf seconds\n", elapsed);
    
    printf("N = %d, K = %d\n", N, K);
    for(int i = 0; i < K; i++)
       printf("Center: (%f,%f) : Size: %d\n", centroides[i].x, centroides[i].y, centroides[i].total_pontos); 
    printf("Iterations: %d\n", n_iter);
    
}
