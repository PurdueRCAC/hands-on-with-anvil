COMP = nvcc
CFLAGS =

INCLUDES  = -I${RCAC_OPENBLAS_ROOT}/include
LIBRARIES = -L${RCAC_OPENBLAS_ROOT}/lib -lopenblas  -L${CUDA_PATH}/lib64 -lcublas

cpu_gpu_dgemm: cpu_gpu_dgemm.o
	${COMP} ${LFLAGS} ${LIBRARIES} cpu_gpu_dgemm.o -o cpu_gpu_dgemm

cpu_gpu_dgemm.o: cpu_gpu_dgemm.cpp
	${COMP} ${CFLAGS} ${INCLUDES} -c cpu_gpu_dgemm.cpp

.PHONY: clean

clean:
	rm -f cpu_gpu_dgemm *.o