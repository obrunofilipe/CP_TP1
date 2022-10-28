CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means
KMEANS = kmeans

CFLAGS = -Wall -g -O2

.DEFAULT_GOAL = k_means

k_means: $(SRC)$(KMEANS).c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)$(KMEANS).c $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*
run:
	./$(BIN)$(EXEC)
