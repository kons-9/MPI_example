#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define N 256
#define DEBUG 1
#define EPS 1.0e-18

double A[N][N], B[N][N];
int myid, numprocs;

int main(int argc, char **argv) {
  int i, j;
  double dc_inv;
  double t0, t1, t2, t_w;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  if (myid == 0) {
    printf("numprocs = %d\n", numprocs);
    if (DEBUG == 1) {
      for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
          A[i][j] = i * j;
          B[i][j] = A[i][j];
        }
      }
    } else {
      srand(1);
      dc_inv = 1.0 / (double)RAND_MAX;
      for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
          A[i][j] = rand() * dc_inv;
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  // transpose
  MPI_Alltoall(&A[0][0], N, MPI_DOUBLE, &B[0][0], N, MPI_DOUBLE,
               MPI_COMM_WORLD);
  MPI_Gather(&B[0][0], N, MPI_DOUBLE, &B[0][0], N, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);

  t2 = MPI_Wtime();
  t0 = t2 - t1;
  MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (myid == 0) {
    printf("N  = %d \n", N);
    printf("Transpose-Mat time  = %lf [sec.] \n", t_w);
  }

  if (DEBUG && myid == 0) {
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        if (A[i][j] != B[j][i]) {
          printf("Error: i,j = %d, %d\n", i, j);
          MPI_Finalize();
          exit(0);
        }
      }
    }
    printf("OK.\n");
  }

  MPI_Finalize();
}
