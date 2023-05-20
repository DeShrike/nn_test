CC = gcc
FLAGS = -O3 -Wall -Wextra
OBJS = nn.o matrix.o
LIBS = -lm

all: xor adder dump_nn

xor.o: xor.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

adder.o: adder.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

dump_nn.o: dump_nn.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

matrix.o: matrix.c matrix.h
	$(CC) $(FLAGS) -c $<

nn.o: nn.c nn.h matrix.h
	$(CC) $(FLAGS) -c $<

xor: xor.o $(OBJS)
	$(CC) xor.o $(OBJS) -o xor $(LIBS)

adder: adder.o $(OBJS)
	$(CC) adder.o $(OBJS) -o adder $(LIBS)

dump_nn: dump_nn.o $(OBJS)
	$(CC) dump_nn.o $(OBJS) -o dump_nn $(LIBS)

clean:
	rm *.o xor adder


