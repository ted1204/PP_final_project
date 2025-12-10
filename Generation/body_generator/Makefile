CC = gcc
CXX = g++
NVCC = nvcc

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61

CCFLAGS = -O2
LDFLAGS = -lm
EXES = testcase1 testcase2 testcase3

all: $(EXES)

testcase3: testcase3.c
	$(CC) $(CCFLAGS) -o $@ $? $(LDFLAGS)

testcase2: testcase2.c
	$(CC) $(CCFLAGS) -o $@ $? $(LDFLAGS)

testcase1: testcase1.c
	$(CC) $(CCFLAGS) -o $@ $? $(LDFLAGS)