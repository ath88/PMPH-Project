# CXX        = g++
CXX        = nvcc
# CXXFLAGS   = -fopenmp -O3 -DWITH_FLOATS=1
CXXFLAGS   = -O3 -DWITH_FLOATS=1 -DCOUNT=1 -lcuda

INCLUDES    += -I ../include
GPU_OPTS   = -arch=sm_20 -D lgWARP=5

SOURCES_CPP =ProjectMain.cpp ProjHelperFun.cpp ProjCoreOrig.cu timers.cu
HELPERS     =ProjHelperFun.h ../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/OpenmpUtil.h timers.h
OBJECTS     =ProjectMain.o ProjHelperFun.o ProjCoreOrig.o timers.o
EXECUTABLE  =runproject


default: gpu

.cpp.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -c -o $@ $<

.c.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -c -o $@ $<

%.o: %.cu $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -c -o $@ $<

gpu: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -o $(EXECUTABLE) $(OBJECTS)


run_tiny: $(EXECUTABLE)
	cat ../Data/Tiny/input.data ../Data/Tiny/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)

