CC = gcc
FLAGS = -O3 -Wall -Wextra
LIBS = -lm

all: test

test.o: test.c ml.h
	$(CC) $(FLAGS) -c $<

ml.o: ml.c ml.h
	$(CC) $(FLAGS) -c $<

test: test.o ml.o
	$(CC) test.o ml.o -o test $(LIBS)
