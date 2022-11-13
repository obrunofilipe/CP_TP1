CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means


CFLAGS = -Wall -g -O2

.DEFAULT_GOAL = kmeansv3

kmeansv1: $(SRC)k_means.c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)k_means.c $(BIN)utils.o -o $(BIN)$(EXEC)

kmeansv2:$(SRC)kmeans.c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)kmeans.c $(BIN)utils.o -o $(BIN)$(EXEC)

kmeansv3: $(SRC)kmeansv3.c
	$(CC) $(CFLAGS) $(SRC)kmeansv3.c -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*
run:
	./$(BIN)$(EXEC)
