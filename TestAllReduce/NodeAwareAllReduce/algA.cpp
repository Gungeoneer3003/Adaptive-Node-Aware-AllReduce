#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <chrono>
#include "header.h"
using namespace std::chrono;

//This is largely based on what I did for 364's Part B to Assignment 3

int AlgA(int n) {
	MPI_Comm topo_comm;
	topo_comm = MPI_COMM_WORLD;

	int rank, nprocs;
	MPI_Comm_rank(topo_comm, &rank); //Rank = id
	MPI_Comm_size(topo_comm, &nprocs); //Count of processes

	auto start = high_resolution_clock::now();

	if (rank == 0) {
		printf("Part B\n");
		start = high_resolution_clock::now();
	}

	int* A = 0, * B = 0, * C = 0;
	int* counts = 0, * displs = 0;
	int base, rem;

	srand(42);
	A = (int*)malloc(n * n * sizeof(int));
	B = (int*)malloc(n * n * sizeof(int));

	//Set up matrices
	if (rank == 0) {
		C = (int*)malloc(n * n * sizeof(int));


		for (int i = 0; i < n * n; i++) {
			A[i] = rand() % 10;
			B[i] = rand() % 10;
		}

		counts = (int*)malloc(nprocs * sizeof(int));
		displs = (int*)malloc(nprocs * sizeof(int));

		base = n / nprocs, rem = n % nprocs;

		int off = 0;
		for (int p = 0; p < nprocs; p++)
		{
			int rows = base + (p < rem);
			counts[p] = rows * n;
			displs[p] = off;
			off += counts[p];
		}
	}

	//Split up the work
	int nloc;
	MPI_Scatter(counts, 1, MPI_INT, &nloc, 1, MPI_INT, 0, topo_comm);

	int* loc = (int*)malloc(nloc * sizeof(int));

	MPI_Scatterv(A, counts, displs, MPI_INT, loc, nloc, MPI_INT, 0, topo_comm);
	MPI_Bcast(B, n * n, MPI_INT, 0, topo_comm);

	//Do the matrix calculation for the sum of the product of two matrices
	int ls = 0, gs;
	for (int i = 0; i < nloc / n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				ls += loc[i * n + k] * B[k * n + j];
			}
		}
	}

	MPI_Allreduce(&ls, &gs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

	if (rank == 0) {
		auto end = high_resolution_clock::now();

		//printf("Matrix A:\n");
		//printMat(A, n);
		//
		//printf("Matrix B:\n");
		//printMat(B, n);
		//
		//printf("Matrix C:\n");
		//printMat(C, n);

		auto elapsed_ns = duration_cast<microseconds>(end - start);
		printf("The sum gotten was %d\n", gs);
		printf("Time taken: %lld microseconds\n", elapsed_ns.count());
	}

	free(loc);
	free(A);
	free(B);

	if (rank == 0) {
		free(C);
		printf("\n");
	}
	return 0;

}