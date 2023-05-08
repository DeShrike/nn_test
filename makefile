CC = gcc
FLAGS = -O3 -Wall -Wextra
OBJS = nn.o matrix.o
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

xor: xor.o $(OBJS)
	$(CC) xor.o $(OBJS) -o xor $(LIBS)

adder: adder.o $(OBJS)
	$(CC) adder.o $(OBJS) -o adder $(LIBS)
