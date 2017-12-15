CC = cc -std=c11
NVCC = nvcc
MPICC = mpicc -std=c11 -lm
HOST := $(shell hostname)
ifeq ($(HOST),cisc372)
RUN = srun -n 1 --gres=gpu:1
MPIRUN = srun
OMPRUN = srun -n 1 -c 40
LIBS = -lgd
FLAGS = -Xcompiler -fopenmp
FFMPEG = ffmpeg
else
# These work for OS X with GD installed by MacPorts.
# edit as needed for your system...
RUN = 
MPIRUN = mpiexec
OMPRUN = 
LIBS = -lgd
FLAGS = -I/opt/local/include -L/opt/local/lib 
FFMPEG = ffmpeg
endif


######################### Parallel Version #########################

cudaonly.exe: lfgpar_cuda.cu Makefile
	$(NVCC) -o cudaonly.exe $(FLAGS) lfgpar_cuda.cu $(LIBS)

cudaonly: cudaonly.exe
	$(RUN) ./cudaonly.exe

hybrid.exe: lfgpar_hybrid.cu Makefile
	$(NVCC) -o hybrid.exe $(FLAGS) lfgpar_hybrid.cu $(LIBS)

hybrid: hybrid.exe
	$(OMPRUN) --gres=gpu:1 ./hybrid.exe

mpionly.exe: lfgpar_mpi.c Makefile
	$(MPICC) -o mpionly.exe lfgpar_mpi.c $(LIBS) -lm

mpionly: mpionly.exe
	$(MPIRUN) -n 3 ./mpionly.exe

seq.exe: lfg.c Makefile
	$(CC) -o seq.exe lfg.c $(LIBS)

seq: seq.exe
	srun -n 1 ./seq.exe

clean:
	rm -f *.mp4 *.exe a.out *.o *~ \#*\#
