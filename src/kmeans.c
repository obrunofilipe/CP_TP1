#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.h"

#define K 4
#define N 10000000





float dist(Ponto point, Centroide centroide){
    //return fabs(point.x - centroide.x) + fabs(point.y - centroide.y);
    return (point.x - centroide.x) * (point.x - centroide.x) 
    + (point.y - centroide.y) * (point.y - centroide.y);
}

int assign_point_to_cluster(Ponto* point, Centroide* centroides, int changed_some_point){
    
    float d = 10000.0f;
    int cluster = -1;

    for(int i = 0; i < K; i++){

        float tmp = dist(*point, centroides[i]);
        if (tmp < d){
            d = tmp;
            cluster = i;
        }
    }
    int cl = get_cluster(*point);
    if (cl != cluster){
        remove_ponto_cluster(*point, &centroides[cl]);
        atualiza_cluster(point, cluster);
        adiciona_ponto_cluster(*point, &centroides[cluster]);
        changed_some_point = 1;
    }

    return changed_some_point;
}

int iterate_points(Ponto* points, Centroide* centroides, int changed_some_point){
    for (int i = 0 ; i < N ; i++){
        changed_some_point = assign_point_to_cluster(&points[i], centroides, changed_some_point);
    }

    return changed_some_point;
}

void new_centroids(Centroide* centroides){
    for(int i = 0; i < K; i++){
        atualiza_centroide(&centroides[i]);
    }
}


void k_means(Ponto* points, Centroide* centroids){

    int changed_some_point = 1;
    int n_iter = 0;

    while(changed_some_point){
        changed_some_point = 0;
        changed_some_point = iterate_points(points, centroids, changed_some_point);
        //printf("ponto alterado = %d\n",changed_some_point);
        new_centroids(centroids);
        n_iter++;
        for(int i = 0; i < K; i++){
            printf("(%f,%f)\n", centroids[i].x,centroids[i].y);
        }
    }
    printf("Número de iterações: %d\n", n_iter);

}


int main(){
    Ponto * pontos;
    Centroide * centroides;
    
    init_pontos(&pontos, &centroides, N,K);

    printf(":: PONTOS ::\n");
    for (int i = 0; i<  N ; i++ ){
        printf("(%f,%f)\n", pontos[i].x, pontos[i].y);
    }

    printf(":: INICIO KMEANS ::\n");

    k_means(pontos,centroides);
    
    for(int i = 0; i < K; i++)
        printf("[%d] soma pontos: (%f,%f) | total pontos: %d\n",i, centroides[i].x, centroides[i].y, centroides[i].total_pontos);


   
}