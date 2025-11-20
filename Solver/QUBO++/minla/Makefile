
CXX = g++

CXXFLAGS_BASE = -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion \
                -Wtype-limits -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual -funroll-loops \
                -std=c++17 -I../include

ifdef DEBUG
  CXXFLAGS = $(CXXFLAGS_BASE) -g
else
  CXXFLAGS = $(CXXFLAGS_BASE) -O3
endif



LD_PATH = ../lib
LDFLAGS = -L$(LD_PATH) -Wl,-rpath,$(LD_PATH)
LIBS = -lqbpp -ltbb -lpthread -lboost_program_options 
LINK_FLAGS = $(LDFLAGS) $(LIBS)

ABS3C32_FLAGS = -lqbpp_abs3c32 -DCOEFF_TYPE=int32_t -DENERGY_TYPE=int64_t
ABS3C64_FLAGS = -lqbpp_abs3c64 -DCOEFF_TYPE=int64_t -DENERGY_TYPE=int64_t
CPP_INT_FLAGS = -DCOEFF_TYPE=cpp_int -DENERGY_TYPE=cpp_int



GRB_DIR = ${HOME}/gurobi
GRB_INC = -I$(GRB_DIR)/linux64/include
GRB_LIB = -L$(GRB_DIR)/linux64/lib -lgurobi_c++ -lgurobi120 -Wl,-rpath,$(GRB_DIR)/linux64/lib

TARGETS = easy_solver easy_warm_start exhaustive

all: $(TARGETS)

easy_solver: easy_solver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LINK_FLAGS)

easy_warm_start: easy_warm_start.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LINK_FLAGS)

exhaustive: exhaustive.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LINK_FLAGS)

clean:
	rm -f $(TARGETS) graph_*.txt
