
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

TARGETS =factorization integer_poly nqueen_easy graph_color tsp labs shift_scheduling factorization_abs3 labs_abs3
GRB_TARGETS = ilp_grb

all: $(TARGETS)

grb: $(GRB_TARGETS)

factorization: factorization.cpp
	$(CXX) $(CXXFLAGS) $(CPP_INT_FLAGS) -o $@ $^ $(LINK_FLAGS) 

integer_poly: integer_poly.cpp
	$(CXX) $(CXXFLAGS) $(CPP_INT_FLAGS) -o $@ $^ $(LINK_FLAGS)
tsp: tsp.cpp
	$(CXX) $(CXXFLAGS) $(CPP_INT_FLAGS) -o $@ $^ $(LINK_FLAGS)

labs: labs.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LINK_FLAGS) 

nqueen_easy: nqueen_easy.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LINK_FLAGS)

graph_color: graph_color.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LINK_FLAGS)

shift_scheduling: shift_scheduling.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LINK_FLAGS) 

factorization_abs3: factorization_abs3.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LINK_FLAGS) $(ABS3C64_FLAGS)

labs_abs3: labs_abs3.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LINK_FLAGS) $(ABS3C32_FLAGS) 

ilp_grb: ilp_grb.cpp
	$(CXX) $(CXXFLAGS) $(GRB_INC) -o $@ $^ $(LINK_FLAGS) $(GRB_LIB)

test: test.cpp
	$(CXX) $(CXXFLAGS) $(CPP_INT_FLAGS) -o $@ $^ $(LINK_FLAGS)

clean:
	rm -f $(TARGETS)
