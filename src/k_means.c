#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000000
#define K 4





float dist(float ponto_x, float ponto_y, float centr_x, float centr_y){
    
    
    return ((ponto_x-centr_x)*(ponto_x-centr_x) )+ ((ponto_y-centr_y)*(ponto_y-centr_y));
    //return rand() / RAND_MAX;
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
        //printf("distancia:: %f | centroide %d\n",tmp,i/2);
        if (tmp < d){
            d = tmp;
            cluster = i/2;
            //printf("cluster %d | i = %d\n", i/2,i );
        }
    }
    if((*points_in_cluster)[point_index] != cluster){ // verificar se o ponto mudou de cluster
        (*points_in_cluster)[point_index] = cluster;
        changed_some_point = 1; //no caso de este ponto mudar de cluster flag passa a true
        //printf("Point changed for cluster %d\n", cluster);
    }else{
        //printf("Point didn't change for cluster %d\n", cluster);
    }
    
    //printf("cluster :: %d\n",cluster);
    //printf("points in cluster :: %d\n", (*points_in_cluster)[point_index]);

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
        /*if ((*points_in_cluster)[i/2] == 0 ){
            printf("%f + %f = %f\n", sum[(*points_in_cluster)[i/2]+1], (*points)[i+1], sum[(*points_in_cluster)[i/2]+1] + (*points)[i+1]);
            printf("Somar ponto (%f,%f) | Soma atual: (%f,%f) | Nº pontos %d\n", (*points)[i], (*points)[i+1], sum[0], sum[1], number_points_cluster[0]);
            printf(":: ITERAÇÂO %d ::", i);
        }
        */ 
       //if ((*points_in_cluster)[i/2] == 0 )
       //printf("sum[1] = %f | i = %d | cluster = %d \n",sum[1],i,(*points_in_cluster)[i/2]);
        sum[2*(*points_in_cluster)[i/2]] += (*points)[i];
        sum[2*(*points_in_cluster)[i/2]+1] +=  (*points)[i+1];
        number_points_cluster[(*points_in_cluster)[i/2]]++;
        //printf("sum[1] = %f | i = %d | cluster = %d \n\n",sum[1],i,(*points_in_cluster)[i/2]);
        
    }
    
    for(int i = 0; i < K*2; i+=2){
        //printf("Soma dos pontos, total de pontos :: (%f,%d)\n", sum[(*points_in_cluster)[i/2]], number_points_cluster[(*points_in_cluster)[i/2]]);
        //printf("Soma dos pontos, total de pontos :: (%f,%d)\n\n", sum[(*points_in_cluster)[i/2]+1], number_points_cluster[(*points_in_cluster)[i/2]]);

        (*centroids)[i] = (float) sum[i] / (float) number_points_cluster[i/2];
        (*centroids)[i+1] = (float) sum[i+1] / (float) number_points_cluster[i/2];
        //printf("Novo centroide :: (%f,%f) | Número de pontos nesse cluster: %d\n", (*centroids)[i], (*centroids)[i+1], number_points_cluster[i/2]);
    }
    
    //for(int i = 0 ; i < K ; i++)
    //    printf("number of points in cluster %d : %d \n",i,number_points_cluster[i]);


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
        /*
        for(int i = 0; i<K*2;i+=2){
            printf("centroids :: %f,%f | \n", (*centroids)[i],(*centroids)[i+1]);
        }
        */
        n_iter++;
        changed_some_point = 0;
    }
    printf("Número de iterações: %d\n", n_iter);
}


int main(){

    float* points;
    float* centroids;
    int* points_in_cluster;
    

    initPoints(&points, &centroids, &points_in_cluster);
    
    printf("points::\n");

    for(int i = 0; i < N*2; i+=2){
        if(points[i] > 1 || points[i+1] > 1)
            printf("(%f,%f)\n",points[i],points[i+1]);
    }
    

    printf("\n");

    
    printf("\nKMEANS::\n");
    k_means(&points, &centroids, &points_in_cluster);
    /*
    for(int i = 0; i < N*2; i+=2){
        if(points[i] > 1 || points[i+1] > 1)
            printf("(%f,%f)\n",points[i],points[i+1]);
    }
*/
    for(int i = 0; i<K*2;i+=2){
        printf("centroids :: %f,%f | \n", centroids[i],centroids[i+1]);
    }
    

    
    
    
}


// (0.000078,0.315378)
// (0.556053,0.586501)
// (0.327672,0.189592)
// (0.470446,0.788647)
// (0.792964,0.346929)
// (0.835021,0.194164)
// (0.309653,0.345721)
// (0.534616,0.297002)
// (0.711494,0.076982)
// (0.834157,0.668422)

// (0.000078,0.315378)
// (0.556053,0.586501)
// (0.327672,0.189592)
// (0.470446,0.788647)