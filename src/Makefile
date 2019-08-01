
CC=gcc
CFLAGS=-O2 -Wall -Wextra
LFLAGS=-lgsl

all: polyc

polyc: polyc.c
	$(CC) $(CFLAGS) -o $@ $@.c $(LFLAGS)

.PHONY: clean
clean:
	rm -f polyc polygroup1_test.txt output_file.txt

