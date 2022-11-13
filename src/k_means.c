#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000000
#define K 4





float dist(float ponto_x, float ponto_y, float centr_x, float centr_y){
    
    return (ponto_x-centr_x)*(ponto_x-centr_x) + (ponto_y-centr_y)*(ponto_y-centr_y);
}


void initPoints(float** points, float** centroids, int** points_in_cluster){

    *points = malloc(sizeof(float)*2*N); //amostras 
    *centroids = malloc(sizeof(float)*K*2); //centroids de cada clusters 
    *points_in_cluster = malloc(sizeof(int)*N); //cluster a que pertence o ponto de índice i

    srand(10); //inicialização dos pontos
    for (int i = 0; i < N*2; i+=2){
        (*points)[i] = (float) rand() / RAND_MAX;
        (*points)[i+1] = (float) rand() / RAND_MAX;
    }

    for(int i = 0; i < N; i++){
        (*points_in_cluster)[i] = -1;
    }
    
    for(int i  = 0; i < K*2 ; i+=2){ // atribuição dos primeiros valores do centroid dos clusters
        (*centroids)[i] = (*points)[i];
        (*centroids)[i+1] = (*points)[i+1];
    }
}



// caso algum ponto mude de cluster -> retorna true (1)
int assign_point_to_cluster(float point_x, float point_y, int point_index, float** centroids, int** points_in_cluster, int changed_some_point){

    float d = 10000.0f;
    int cluster = -1;
    for(int i = 0; i < K*2; i+=2){
        float tmp = dist(point_x, point_y, (*centroids)[i], (*centroids)[i+1]);
        if (tmp < d){
            d = tmp;
            cluster = i/2;
        }
    }
    if((*points_in_cluster)[point_index] != cluster){ // verificar se o ponto mudou de cluster
        (*points_in_cluster)[point_index] = cluster;
        changed_some_point = 1; //no caso de este ponto mudar de cluster flag passa a true
    }

    return changed_some_point;
}

int iterate_points(float** points, float** centroids, int** points_in_cluster, int changed_some_point){
    for(int i = 0; i < N*2; i+= 2){
        changed_some_point = assign_point_to_cluster((*points)[i], (*points)[i+1], i/2, centroids, points_in_cluster, changed_some_point);
    }
    return changed_some_point;
}


void new_centroids(float** points, float** centroids, int** points_in_cluster){

    float* sum = malloc(sizeof(float)*K*2);
    int* number_points_cluster = malloc(sizeof(int)*K);

    for(int i = 0; i < K*2; i+=2){
        sum[i] = 0.0f;
        sum[i+1] = 0.0f;
        number_points_cluster[i/2] = 0;
    }

    for(int i = 0; i < N*2; i+=2){
        sum[2*(*points_in_cluster)[i/2]] += (*points)[i];
        sum[2*(*points_in_cluster)[i/2]+1] +=  (*points)[i+1];
        number_points_cluster[(*points_in_cluster)[i/2]]++;
    }
    
    for(int i = 0; i < K*2; i+=2){
        (*centroids)[i] = (float) sum[i] / (float) number_points_cluster[i/2];
        (*centroids)[i+1] = (float) sum[i+1] / (float) number_points_cluster[i/2];
    }

    free(sum);
    free(number_points_cluster);
}


void k_means(float** points, float** centroids, int** points_in_cluster){
    int changed_some_point = 1;
    int n_iter = 0;
    while (changed_some_point){
        changed_some_point = 0;
        changed_some_point = iterate_points(points, centroids, points_in_cluster, changed_some_point);
        new_centroids(points, centroids, points_in_cluster);
        n_iter++;

    }
    printf("Número de iterações: %d\n", n_iter);
}


int main(){

    float* points;
    float* centroids;
    int* points_in_cluster;
    

    initPoints(&points, &centroids, &points_in_cluster);
        
    
    printf("\nKMEANS::\n");
    k_means(&points, &centroids, &points_in_cluster);

    for(int i = 0; i<K*2;i+=2){
        printf("centroids :: %f,%f | \n", centroids[i],centroids[i+1]);
    }   
}