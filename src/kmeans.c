#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.h"

#define K 4
#define N 10000000


float dist(Ponto point, Centroide centroide){
    float ponto_x = get_x_point(point);
    float ponto_y = get_y_point(point);
    float centroide_x = get_x_centroide(centroide);
    float centroide_y = get_y_centroide(centroide);
    
    return (ponto_x - centroide_x) * (ponto_x- centroide_x) 
    + (ponto_y - centroide_y) * (ponto_y - centroide_y);
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
    for(int i = 0; i < K; i++){ // para cada centroide, vai ser calculado o novo valor das suas coordenadas, com base na média geométrica dos pontos que estão no cluster correspondente
        atualiza_centroide(&centroides[i]);
    }
}


void k_means(Ponto* points, Centroide* centroids){

    int changed_some_point = 1;
    int n_iter = 0;

    while(changed_some_point){ // enquanto há pontos a mudarem de cluster
        changed_some_point = 0;
        changed_some_point = iterate_points(points, centroids, changed_some_point);
        new_centroids(centroids);
        n_iter++;
    }
    printf("Número de iterações: %d\n", n_iter);
}


int main(){
    Ponto * pontos;
    Centroide * centroides;
    
    init_pontos(&pontos, &centroides, N,K);


    k_means(pontos,centroides);
    
    for(int i = 0; i < K; i++)
       printf("[%d] soma pontos: (%f,%f) | total pontos: %d\n",i, centroides[i].x, centroides[i].y, centroides[i].total_pontos); 

}