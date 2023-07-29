#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define N 1152
// #PJM --mpi proc=576
// #define N 15
#define DEBUG 1
#define EPS 1.0e-18

double A[N][N];
double B[N][N];

int myid, numprocs;

void solve(int n) {
  // processes comunicate by MPI_gather, MPI_Scatter
  // promise: n / numprocss = n_local (integer)
  int i, j;
  int ierr;
  int n_local = n / numprocs;

  double *matALocal = (double *)malloc(sizeof(double) * n * n_local);

  // initialize: each process has (block, *) matrix
  ierr = MPI_Scatter(&A[0][0], n * n_local, MPI_DOUBLE, matALocal, n * n_local,
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // transpose block
  double *matALocalTrans = (double *)malloc(sizeof(double) * n * n_local);
  for (i = 0; i < n_local; i++) {
    for (j = 0; j < n; j++) {
      matALocalTrans[j * n_local + i] = matALocal[i * n + j];
    }
  }

  // finalize: each process has (*, *) matrix

  for (i = 0; i < n; i++) {
    ierr = MPI_Gather(&matALocalTrans[i * n_local], n_local, MPI_DOUBLE,
                      &A[i][0], n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  free(matALocal);
  free(matALocalTrans);
  return;
}

int main(int argc, char *argv[]) {

  int ierr, rc;
  int i, j;
  double dc_inv;
  double t0, t1, t2, t_w;

  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  if (myid == 0) {
    printf("numprocs = %d\n", numprocs);
    if (DEBUG == 1) {
      srand(1);
      dc_inv = 1.0 / (double)RAND_MAX;
      for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
          A[i][j] = rand() * dc_inv;
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

  ierr = MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  solve(N);

  t2 = MPI_Wtime();
  t0 = t2 - t1;
  ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (myid == 0) {
    printf("N  = %d \n", N);
    printf("Mat-Mat time  = %lf [sec.] \n", t_w);

    double d_mflops = 2.0 * (double)N * (double)N / t_w;
    d_mflops = d_mflops * 1.0e-6;
    printf(" %lf [MFLOPS] \n", d_mflops);
  }

  if (DEBUG && myid == 0) {
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        if (A[i][j] != B[j][i]) {
          printf("Error: i,j = %d, %d\n", i, j);
          rc = MPI_Finalize();
          exit(0);
        }
      }
    }
    printf("OK.\n");
  }

  rc = MPI_Finalize();
  exit(0);
}
