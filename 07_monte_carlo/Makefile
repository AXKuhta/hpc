all: sequential.elf

sequential.elf: sequential.c
	gcc -O2 -Wall -Wextra sequential.c -o sequential.elf

parallel.elf: parallel.cu
	nvcc -O2 -ccbin /usr/bin/g++-10 -Xcompiler -Wall,-Wextra parallel.cu -o parallel.elf # -lcurand
