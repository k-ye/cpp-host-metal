# Terribly handwritten makefile, use cmake
CC=clang++
CFLAGS=-std=c++17 -O2 -Wall -Wno-unused-function
LDFLAGS=-framework Metal -framework CoreGraphics -framework Foundation
BUILDDIR=build
OBJDIR=$(BUILDDIR)
OBJ=$(OBJDIR)/main.o
BINARY=mtl.app


all: $(OBJ)
	$(CC) -o $(BINARY) $(OBJ) $(LDFLAGS)


$(OBJDIR)/%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(BUILDDIR)/* $(BINARY)