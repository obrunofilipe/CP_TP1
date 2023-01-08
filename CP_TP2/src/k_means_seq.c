#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


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



void init_pontos(Ponto** p, Centroide** c){
     
    *p = malloc(sizeof( Ponto ) * N);
    *c = malloc(sizeof( struct centroide ) * K);
    
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


float dist(Ponto point, Centroide centroide){


    return (point.x - centroide.x) * (point.x- centroide.x) 
    + (point.y - centroide.y) * (point.y - centroide.y);
}

int assign_point_to_cluster(Ponto* point, Centroide* centroides, int changed_some_point){
    
    float d = 10000.0f;
    int cluster = -1;

    for(int j = 0; j < K; j++){
        
        float tmp = dist(*point, centroides[j]);
        if (tmp < d){
            d = tmp;
            cluster = j; 
        }
    }


    int cl = point->cluster; // obtençao do cluster em que estava o ponto
    
    //printf("d: %f | cluster: %d | cl: %d\n", d, cluster, cl);
    
    if (cl != cluster){ // se o cluster em que estava o ponto for diferente do cluster em que vai ser inserido agora, temos de alterar as estruturas do ponto e dos centroides envolvidos
        //remover ponto do cluster

        if(cl != -1){
            centroides[cl].soma_x -= point->x;
            centroides[cl].soma_y -= point->y;
            centroides[cl].total_pontos--;
        }
        

        
        //atualizar cluster do ponto
        point->cluster = cluster;

        // adicionar ponto ao cluster

        
        centroides[cluster].soma_x += point->x;
        centroides[cluster].soma_y += point->y;
        centroides[cluster].total_pontos++;

        changed_some_point = 1; // pelo menos um ponto foi alterado
    }
    

    return changed_some_point;
}


int iterate_points(Ponto* points, Centroide* centroides, int changed_some_point){
    for (int i = 0 ; i < N ; i++){ // para todos os pontos, é efetuado o cálculo da distância do ponto a todos os centroides
        changed_some_point = assign_point_to_cluster(&points[i], centroides, changed_some_point);
        //printf("Changed_some_point: %d", changed_some_point);
    }

    return changed_some_point;
}

void new_centroids(Centroide* centroides){// para cada centroide, vai ser calculado o novo valor das suas coordenadas, com base na média geométrica dos pontos que estão no cluster correspondente
    for(int i = 0; i < K; i++){
        centroides[i].x = centroides[i].soma_x / centroides[i].total_pontos;
        centroides[i].y = centroides[i].soma_y / centroides[i].total_pontos;
    }
}


int k_means(Ponto* points, Centroide* centroids){

    int changed_some_point = 1;
    int n_iter = 0;

    while(changed_some_point && n_iter < 20){// enquanto há pontos a mudarem de cluster
        changed_some_point = 0;
        changed_some_point = iterate_points(points, centroids, changed_some_point);
        new_centroids(centroids);
        n_iter++;
    }
    return n_iter;
}


int main(int argc, char** argv){
    Ponto * pontos;
    Centroide * centroides;
    
    N = atoi(argv[1]);
    K = atoi(argv[2]);
    

    

    
    
    init_pontos(&pontos, &centroides);


    int n_iter = k_means(pontos,centroides);
    
    printf("N = %d, K = %d\n", N, K);
    for(int i = 0; i < K; i++)
       printf("Center: (%f,%f) : Size: %d\n", centroides[i].x, centroides[i].y, centroides[i].total_pontos); 
    printf("Iterations: %d\n", n_iter);
    
}
