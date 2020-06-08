# Compile serialization test

# Relative path of this directory to the source
RELPATH      = ../..

WARNING      = -Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast
COMPILECPP   = g++ -std=c++17 -g -O0 ${WARNING}
COMPILENVCC  = nvcc -std=c++11 -dc
COMPILEEXEC  = nvcc

# Main program
MAIN_PRG     = gpu_test.cpp
# Bidirected graph sources
BG_SRCS      = ${RELPATH}/src/BidirectedGraph.cpp ${RELPATH}/src/BidirectedGraphGPU.cpp
# GPU algorithm sources
ALGOGPU_SRCS = ${RELPATH}/src/algorithms/find_sssp.cpp 
# Handlegraph sources
HG_SRCS      = ${RELPATH}/deps/libhandlegraph/src/handle.cpp
# JSON library sources
JSON_SRCS    = ${RELPATH}/deps/jsoncpp/dist/jsoncpp.cpp 
# Compiled sources and objects
SOURCES      = ${MAIN_PRG} ${BG_SRCS} ${HG_SRCS} ${JSON_SRCS}
OBJECTS      = ${SOURCES:.cpp=.o}
GPU_SOURCES  = ${ALGOGPU_SRCS}
GPU_OBJECTS  = ${GPU_SOURCES:.cpp=.o}
# Executable binary
EXECBIN      = SSSP.exe 

all : ${EXECBIN}

${EXECBIN} : ${OBJECTS} ${GPU_OBJECTS}
	${COMPILEEXEC} ${OBJECTS} ${GPU_OBJECTS} -o ${EXECBIN}

${OBJECTS}: ${SOURCES}
	${COMPILECPP} -c $^ -o $@

${GPU_OBJECTS}: ${GPU_SOURCES}
	${COMPILENVCC} $^ -o $@

# Removes all intermediate object files but keeps the executable binary
clean :
	- rm ${OBJECTS} ${GPU_OBJECTS}

# Removes all generated files including the executable binary
spotless : clean
	- rm ${EXECBIN}
