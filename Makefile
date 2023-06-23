CC=g++
LINKER_DIRS=-L/usr/local/cuda/lib
LINKER_FLAGS=-lcudart -lcuda
NVCC=nvcc
CUDA_ARCHITECTURE=20
OCELOT=`OcelotConfig -l`

all: cuda1 cuda2 #main_cpu

# main_cpu: main.o
# 	$(CC) main.o -o main_cpu $(LINKER_DIRS) $(OCELOT)

cuda1: main.o cuda1.o
	$(CC) main.o cuda1.o -o cuda1 $(LINKER_DIRS) $(OCELOT)

cuda2: main.o cuda2.o
	$(CC) main.o cuda2.o -o cuda2 $(LINKER_DIRS) $(OCELOT)

main.o: main.cu
	$(NVCC) main.cu -c -I .

cuda1.o: cuda1.cu
	$(NVCC) -c cuda1.cu -arch=sm_$(CUDA_ARCHITECTURE) -I .

cuda2.o: cuda2.cu
	$(NVCC) -c cuda2.cu -arch=sm_$(CUDA_ARCHITECTURE) -I .

clean:
	rm -f main.o kernel-times.json cuda1 cuda2 cuda1.o cuda2.o
