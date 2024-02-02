# Makefile for Neural Network Inference

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -std=c99

# Source files
COMMON_SRC = Bmp2Matrix.c
MAIN_SRC = main.c
PREPROCESS_SRC = preprocess.c

# Output executable
OUT = neural_network_inference

# Default target
all: $(OUT)

ifdef INCLUDE_PREPROCESS
SRC = $(COMMON_SRC) $(PREPROCESS_SRC)
else
SRC = $(MAIN_SRC) $(COMMON_SRC)
endif

# Compile the source files into the executable
$(OUT): $(SRC)
	$(CC) $(CFLAGS) -o $(OUT) $(SRC) -lm

# Run the program with a specific photo as an argument
run: $(OUT)
	./$(OUT) $(ARGS)
	
# Clean up generated files
clean:
	rm -f $(OUT)

# Phony target to avoid conflicts with file names
.PHONY: all run clean
