OBJ=main.cpp
FUNC=
CXX=g++-4.9
CXXOPTIMFLAGS=-O3 
CXXDEBUGFLAGS=-g -O0 -rdynamic
LIBS=../../libmda/lib/libmda.a


gcc: $(OBJ) $(FUNC)
	$(CXX) $(CXXOPTIMFLAGS) -fexpensive-optimizations -Drestrict= -std=c++11 -o main $(OBJ) $(FUNC) $(LIBS)
	./main

gcc_debug: $(OBJ) $(FUNC)
	$(CXX) $(CXXDEBUGFLAGS) -std=c++11 -o main $(OBJ) $(FUNC) $(LIBS)
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./main

gcc_assembly: $(OBJ) $(FUNC)
	$(CXX) $(CXXDEBUGFLAGS) -O3 -S -std=c++11 $(OBJ) $(FUNC)

intel: $(OBJ) $(FUNC)
	icc $(CXXOPTIMFLAGS) -ip -ipo -fast -std=c++11 -Wcontext-limit=100 -o main $(OBJ) $(FUNC)
	./main

intel_debug: $(OBJ) $(FUNC)
	icc $(CXXDEBUGFLAGS) -std=c++11 -Wcontext-limit=100 -o main $(OBJ) $(FUNC)
	valgrind --tool=memcheck --leak-check=yes ./main

clang: $(OBJ) $(FUNC)
	clang++ $(CXXOPTIMFLAGS) -std=c++11 -o main $(OBJ) $(FUNC)
	./main

clang_debug: $(OBJ) $(FUNC)
	clang++ $(CXXDEBUGFLAGS) -std=c++11 -o main $(OBJ) $(FUNC)
	valgrind --tool=memcheck --leak-check=yes ./main

clang_assembly: $(OBJ) $(FUNC)
	clang++ $(CXXDEBUGFLAGS) -O3 -S -std=c++11 $(OBJ) $(FUNC)

clean:
	rm -f *core main
