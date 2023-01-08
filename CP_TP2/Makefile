CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC_SEQ = k_means_seq
EXEC_PAR = k_means_par
THREADS = 20 

CFLAGS = -Wall -g -O2
CFLAGS_PAR = -fopenmp -lm

.DEFAULT_GOAL = runpar

kmeans_seq: $(SRC)k_means_seq.c
	$(CC) $(CFLAGS) $(SRC)k_means_seq.c -o $(BIN)$(EXEC_SEQ)

kmeans_par: $(SRC)k_means_par.c
	$(CC) $(CFLAGS) $(CFLAGS_PAR) $(SRC)k_means_par.c -o $(BIN)$(EXEC_PAR)

clean:
	rm -r bin/*

runseq: kmeans_seq
	./$(BIN)$(EXEC_SEQ) 10000000 $(CP_CLUSTERS)

runpar: kmeans_par
	./$(BIN)$(EXEC_PAR) 10000000 $(CP_CLUSTERS) $(THREADS)

