#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define N 1152
// #PJM --mpi proc=576
// #define N 15
#define DEBUG 1
#define EPS 1.0e-18

int myid, numprocs;

double *solve(double *A_local, double *x_local, int n, int n_local) {
  // A_local: (n, n_local)
  // x_local: (n_local)
  // promise: n / numprocs = n_local (integer)
  // return: y_local: (n_local)

  double *y_local = (double *)malloc(sizeof(double) * n_local);

  double *x = (double *)malloc(sizeof(double) * n);
  int ierr = MPI_Allgather(x_local, n_local, MPI_DOUBLE, x, n_local, MPI_DOUBLE,
                           MPI_COMM_WORLD);

  for (int i = 0; i < n_local; i++) {
    y_local[i] = 0.0;
    int tmp_i = i * n;
    for (int j = 0; j < n; j++) {
      y_local[i] += A_local[tmp_i + j] * x[j];
    }
  }
  // free(x);

  return y_local;
}

int main(int argc, char *argv[]) {

  int ierr, rc;
  int i, j;
  double dc_inv;
  double t0, t1, t2, t_w;

  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  int n_local = N / numprocs;

  double *A_local = (double *)malloc(sizeof(double) * N * n_local);
  double *x_local = (double *)malloc(sizeof(double) * n_local);

  if (DEBUG == 1) {
    for (i = 0; i < n_local; i++) {
      for (j = 0; j < N; j++) {
        A_local[i * N + j] = 1;
      }
      x_local[i] = 1;
    }
  } else {
    srand(1);
    dc_inv = 1.0 / (double)RAND_MAX;
    for (i = 0; i < n_local; i++) {
      for (j = 0; j < N; j++) {
        A_local[i * N + j] = rand() * dc_inv;
      }
      x_local[i] = rand() * dc_inv;
    }
  }

  ierr = MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  double *y_local = solve(A_local, x_local, N, n_local);

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
  if (DEBUG == 1) {
    for (i = 0; i < n_local; i++) {
      if (fabs(y_local[i] - N) > EPS) {
        printf("Error: y[%d] = %lf \n", i, y_local[i]);
        rc = MPI_Finalize();
        exit(0);
      }
    }
    printf("OK! \n");
  }

  rc = MPI_Finalize();
  exit(0);
}
