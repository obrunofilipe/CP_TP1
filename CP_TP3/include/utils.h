#ifndef UTILS_H_
#define UTILS_H_

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


void init_pontos(Ponto**, Centroide**, int, int);
void remove_ponto_cluster(Ponto, Centroide *);
void adiciona_ponto_cluster(Ponto, Centroide *);
void atualiza_cluster(Ponto*,int);
void atualiza_centroide( Centroide * );
float get_x_point(Ponto);
float get_y_point(Ponto);
float get_x_centroide(Centroide);
float get_y_centroide(Centroide);
int get_cluster(Ponto);
void atualiza_centroide(Centroide *);
#endif //UTILS_H_