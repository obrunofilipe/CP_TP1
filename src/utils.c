#include <stdlib.h>
#include <stdio.h>
#include "../include/utils.h"


void init_pontos(Ponto** p, Centroide** c, int N, int K){
    
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
        (*c)[i].soma_x += (*p)[i].x;
        (*c)[i].soma_y += (*p)[i].y;
        (*c)[i].total_pontos = 1;
    }
}



void remove_ponto_cluster(Ponto p, Centroide * c){
    c->soma_x -= p.x;
    c->soma_y -= p.y;
    c->total_pontos--;
}

void adiciona_ponto_cluster(Ponto p, Centroide * c){
    c->soma_x += p.x;
    c->soma_y += p.y;
    c->total_pontos++;
}

float get_x_point(Ponto p){
    return p.x;
}

float get_y_point(Ponto p){
    return p.y;
}

float get_x_centroide(Centroide c){
    return c.x;
}

float get_y_centroide(Centroide c){
    return c.y;
}

int get_cluster(Ponto p){
    return p.cluster;
}

void atualiza_cluster(Ponto*p , int cluster){
    p->cluster = cluster;
}

void atualiza_centroide(Centroide * c){
    c->x = c->soma_x/(float)c->total_pontos;
    c->y = c->soma_y/(float)c->total_pontos;
}
