# Compile object files for code

WARNING    = -Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast
COMPILECPP = g++ -std=c++17 -g -O0 ${WARNING}
MAKEDEPCPP = g++ -std=c++17 -MM

BG_SRCS    = src/BidirectedGraph.cpp
HG_SRCS    = deps/libhandlegraph/src/handle.cpp
JSON_SRCS  = deps/jsoncpp/dist/jsoncpp.cpp
SOURCES    = ${BG_SRCS} ${HG_SRCS} ${JSON_SRCS}
OBJECTS    = ${SOURCES:.cpp=.o}

# Compiles all source files
sources: ${OBJECTS}

build_deps: build_json build_hg

build_json:
	cd deps/jsoncpp && python3 amalgamate.py

build_hg:
	cd deps/libhandlegraph && mkdir build && cd build && cmake .. && make && make install

%.o: %.cpp
	${COMPILECPP} -c $< -o $@

# Removes all compiled code
clean:
	- rm ${OBJECTS}