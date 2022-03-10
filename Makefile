# ccomp?=gcc
ccomp?=nvcc
# cflags=-fopenmp -O3
cflags=-ccbin $(CUDA_GCC_DIR) --compiler-options="-fopenmp -O3"
cexec=many-small-permutes.x
csources=many-small-permutes.cu
cobjects=${patsubst %.cu, %.cu.obj, $(csources)}
clibs=-L $(LIBRARY_PATH) -lcudart -lcuda -lcutt

main: $(cexec)

$(cexec): $(cobjects)
	$(ccomp) $(clibs) -o $(cexec) $(cobjects)

%.cu.obj: %.cu
	$(ccomp) $(cflags) -g -c -o $@ $<

clean:
	rm *.o *.obj $(cexec)

