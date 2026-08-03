# Build the STRATUM engine.
#
# Nothing here is required to use STRATUM. The Python toolchain does not
# call any of it. Build it only if you want the measurement tools or intend
# to work on the streaming engine.
#
# Any C99 compiler works and there are no dependencies.

CC      ?= cc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wshadow
LDFLAGS ?=

SRC   := src
BUILD := build
BIN   := bin

# Windows needs the process information library for the resident memory
# figure. Everything else reads it from the kernel directly.
ifeq ($(OS),Windows_NT)
    LDLIBS += -lpsapi
    EXE := .exe
else
    EXE :=
endif

CORE_SRC := $(SRC)/map.c $(SRC)/gguf.c $(SRC)/direct.c
CORE_OBJ := $(CORE_SRC:$(SRC)/%.c=$(BUILD)/%.o)

TOOLS := stratum-bandwidth stratum-gguf
TESTS := test_gguf

.PHONY: all test clean

all: $(TOOLS:%=$(BIN)/%$(EXE))

$(BUILD)/%.o: $(SRC)/%.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN)/stratum-bandwidth$(EXE): $(BUILD)/bandwidth.o $(CORE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN)/stratum-gguf$(EXE): $(BUILD)/ggufdump.o $(CORE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN)/test_gguf$(EXE): $(BUILD)/test_gguf.o $(CORE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/test_gguf.o: tests/test_gguf.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -I$(SRC) -c $< -o $@

test: $(BIN)/test_gguf$(EXE)
	./$(BIN)/test_gguf$(EXE)

clean:
	rm -rf $(BUILD) $(BIN)
