# quicktok Makefile
#
# Targets:
#   make            -- build the static and shared libraries (default)
#   make lib        -- build build/libquicktok.{a,so/dylib}
#   make test       -- build and run the correctness test (exact vs reference ids)
#   make example    -- build the hello example
#   make install    -- install header + libs + data + pkg-config to $(PREFIX)
#   make clean      -- remove build artefacts
#
# Variables (override on the command line):
#   CXX=clang++     -- C++ compiler (default: c++)
#   PREFIX=...      -- install root (default: /usr/local)
#   DESTDIR=...     -- staging root prepended to install paths

CXX     ?= c++
PREFIX  ?= /usr/local
BUILD   := build

VERSION   := 0.1.0
SOVERSION := 0

# aarch64 (Apple Silicon, Graviton) gets NEON from baseline ARMv8-A and the
# 2-byte-radix trie / dense memo. x86_64 builds correctly with the scalar +
# SSE2 paths; the -DTRIE2/-DIVDENSE wins are validated on aarch64 (see README).
UNAME_M := $(shell uname -m)
ifneq (,$(filter $(UNAME_M),arm64 aarch64))
  CXXFLAGS_ARCH := -mcpu=native -DTRIE2 -DIVDENSE
else
  CXXFLAGS_ARCH := -march=native
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SOEXT        := dylib
  LIB_SO_REAL  := $(BUILD)/libquicktok.$(VERSION).$(SOEXT)
  LIB_SO       := $(BUILD)/libquicktok.$(SOEXT)
  SHARED_FLAGS := -dynamiclib -install_name @rpath/libquicktok.$(SOVERSION).$(SOEXT)
else
  SOEXT        := so
  LIB_SO_REAL  := $(BUILD)/libquicktok.$(SOEXT).$(VERSION)
  LIB_SO       := $(BUILD)/libquicktok.$(SOEXT)
  SHARED_FLAGS := -shared -Wl,-soname,libquicktok.$(SOEXT).$(SOVERSION)
endif

CXXFLAGS ?= -O3 -std=c++20
CXXFLAGS += $(CXXFLAGS_ARCH) -fPIC -Iinclude -Isrc $(CXXFLAGS_EXTRA)
DATADIR  := $(PREFIX)/share/quicktok

SRCS := src/quicktok.cpp src/trie2_mb.cpp
OBJS := $(BUILD)/quicktok.o $(BUILD)/trie2_mb.o

.PHONY: all lib test example install clean
all: lib
lib: $(BUILD)/libquicktok.a $(LIB_SO)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/libquicktok.a: $(OBJS)
	ar rcs $@ $(OBJS)

$(LIB_SO): $(OBJS)
	$(CXX) $(CXXFLAGS) $(SHARED_FLAGS) -o $(LIB_SO_REAL) $(OBJS)
	ln -sf $(notdir $(LIB_SO_REAL)) $(LIB_SO)

$(BUILD):
	mkdir -p $(BUILD)

test: lib
	$(CXX) $(CXXFLAGS) test/test_quicktok.cpp $(BUILD)/libquicktok.a -o $(BUILD)/test_quicktok
	$(BUILD)/test_quicktok data

example: lib
	$(CXX) $(CXXFLAGS) examples/hello.cpp $(BUILD)/libquicktok.a -o $(BUILD)/hello
	$(BUILD)/hello

install: lib
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(DATADIR)
	install -m644 include/quicktok.hpp $(DESTDIR)$(PREFIX)/include/
	install -m644 $(BUILD)/libquicktok.a $(DESTDIR)$(PREFIX)/lib/
	cp -a $(LIB_SO_REAL) $(LIB_SO) $(DESTDIR)$(PREFIX)/lib/
	install -m644 data/cl100k.vocab data/uniclass.bin data/uniclass.bin.meta $(DESTDIR)$(DATADIR)/

clean:
	rm -rf $(BUILD)
