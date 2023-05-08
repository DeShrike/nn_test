CC = gcc
FLAGS = -O3 -Wall -Wextra
OBJS = test.o nn.o matrix.o
LIBS = -lm

all: xor adder

xor.o: xor.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

adder.o: adder.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

matrix.o: matrix.c matrix.h
	$(CC) $(FLAGS) -c $<

nn.o: nn.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

xor: $(OBJS)
	$(CC) $(OBJS) -o xor $(LIBS)

adder: $(OBJS)
	$(CC) $(OBJS) -o adder $(LIBS)
