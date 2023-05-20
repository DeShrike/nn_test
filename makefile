CC = gcc
FLAGS = -O3 -Wall -Wextra
OBJS = nn.o matrix.o
LIBS = -lm

all: xor adder img_nn

xor.o: xor.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

adder.o: adder.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

img_nn.o: img_nn.c nn.h matrix.h image.h
	$(CC) $(FLAGS) -c $<

matrix.o: matrix.c matrix.h
	$(CC) $(FLAGS) -c $<

image.o: image.c image.h
	$(CC) $(FLAGS) -c $<

nn.o: nn.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

xor: xor.o $(OBJS)
	$(CC) xor.o $(OBJS) -o xor $(LIBS)

adder: adder.o $(OBJS)
	$(CC) adder.o $(OBJS) -o adder $(LIBS)

img_nn: img_nn.o image.o $(OBJS)
	$(CC) img_nn.o image.o $(OBJS) -o img_nn $(LIBS) -lpng

clean:
	rm *.o xor adder img_nn


