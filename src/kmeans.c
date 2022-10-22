#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100
#define K 4

void initPoints(){

    float* points = malloc(sizeof(float)*2*N);
    float* centroids = malloc(sizeof(float)*K*2);

    srand(22);
    for (int i = 0; i < N; i += 2){
        points[i] = (float) rand() / RAND_MAX;
        points[i+1] = (float) rand() / RAND_MAX;
    }
    
    for(int i  = 0; i < K ; i+2){
        centroids[i] = points[i];
        centroids[i+1] = points[i+1];
    }
}

int main(){
    
}