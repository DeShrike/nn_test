CC = gcc
FLAGS = -O3 -Wall -Wextra
OBJS = test.o nn.o matrix.o
LIBS = -lm

all: test

test.o: test.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

matrix.o: matrix.c matrix.h
	$(CC) $(FLAGS) -c $<

nn.o: nn.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

test: $(OBJS)
	$(CC) $(OBJS) -o test $(LIBS)
