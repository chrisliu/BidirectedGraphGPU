# Compile serialization test

# Relative path of this directory to the source
RELPATH    = ../..

WARNING    = -Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast
COMPILECPP = g++ -std=c++17 -g -O0 ${WARNING}

# Main program
MAIN_PRG   = gpu_test.cpp
# Bidirected graph sources
BG_SRCS    = ${RELPATH}/src/BidirectedGraph.cpp ${RELPATH}/src/BidirectedGraphGPU.cpp
# Handlegraph sources
HG_SRCS    = ${RELPATH}/deps/libhandlegraph/src/handle.cpp
# JSON library sources
JSON_SRCS  = ${RELPATH}/deps/jsoncpp/dist/jsoncpp.cpp 
# Compiled sources and objects
SOURCES    = ${MAIN_PRG} ${BG_SRCS} ${HG_SRCS} ${JSON_SRCS}
OBJECTS    = ${SOURCES:.cpp=.o}
# Executable binary
EXECBIN    = GPUTest.exe 

all : ${EXECBIN}

${EXECBIN} : ${OBJECTS}
	${COMPILECPP} -o${EXECBIN} ${OBJECTS}

%.o : %.cpp
	${COMPILECPP} -c $< -o $@

# Removes all intermediate object files but keeps the executable binary
clean :
	- rm ${OBJECTS}

# Removes all generated files including the executable binary
spotless : clean
	- rm ${EXECBIN}
