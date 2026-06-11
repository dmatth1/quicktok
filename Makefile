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

VERSION   := 0.3.0
SOVERSION := 0

# The 2-byte-radix trie and dense memo are always on (validated on both arches).
# Default tunes for the build host; override CXXFLAGS_ARCH for portable binaries
# (e.g. CXXFLAGS_ARCH="-march=x86-64-v3" for distributable x86 builds).
UNAME_M := $(shell uname -m)
ifneq (,$(filter $(UNAME_M),arm64 aarch64))
  CXXFLAGS_ARCH ?= -mcpu=native
else
  CXXFLAGS_ARCH ?= -march=native
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

CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra
CXXFLAGS += $(CXXFLAGS_ARCH) -fPIC -Iinclude -Isrc $(CXXFLAGS_EXTRA)
DATADIR  := $(PREFIX)/share/quicktok

SRCS := src/quicktok.cpp src/trie2_mb.cpp src/cabi.cpp
OBJS := $(BUILD)/quicktok.o $(BUILD)/trie2_mb.o $(BUILD)/cabi.o

.PHONY: all lib test example bench bench-py install clean
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
	$(CC) -O2 -Iinclude -c test/test_cabi.c -o $(BUILD)/test_cabi.o
	$(CXX) $(BUILD)/test_cabi.o $(BUILD)/libquicktok.a -o $(BUILD)/test_cabi
	$(BUILD)/test_cabi

example: lib
	$(CXX) $(CXXFLAGS) examples/hello.cpp $(BUILD)/libquicktok.a -o $(BUILD)/hello
	$(BUILD)/hello

bench: lib
	$(CXX) $(CXXFLAGS) bench/bench.cpp $(BUILD)/libquicktok.a -o $(BUILD)/bench
	$(BUILD)/bench data bench/corpus.txt

bench-py:
	python3 bench/bench.py

$(BUILD)/quicktok.pc: quicktok.pc.in | $(BUILD)
	sed 's|@PREFIX@|$(PREFIX)|' quicktok.pc.in > $@

install: lib $(BUILD)/quicktok.pc
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig $(DESTDIR)$(DATADIR)
	install -m644 include/quicktok.hpp include/quicktok.h $(DESTDIR)$(PREFIX)/include/
	install -m644 $(BUILD)/libquicktok.a $(DESTDIR)$(PREFIX)/lib/
	cp -a $(LIB_SO_REAL) $(LIB_SO) $(DESTDIR)$(PREFIX)/lib/
	install -m644 $(BUILD)/quicktok.pc $(DESTDIR)$(PREFIX)/lib/pkgconfig/
	install -m644 data/*.vocab data/*.special data/*.bin data/*.meta $(DESTDIR)$(DATADIR)/

clean:
	rm -rf $(BUILD)
