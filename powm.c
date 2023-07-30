#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define DEBUG 0
#define N 576 * 10
#define MAX_ITER 100
#define EPS 1e-16

/* Please define the matrices in HERE. */
static double A[N][N];

int myid, numprocs;

void MyMatVec(double *y_local, double *A_local, double *x, int n, int n_local);
double PowM(double *A_local, double *x_local, double *y_local, int n,
            int n_local, int *n_iter);

void print_matrix(double *A_local, int n, int n_local) {
  int i, j;
  for (i = 0; i < n_local; i++) {
    for (j = 0; j < n; j++) {
      printf("%lf ", A_local[i * n + j]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {

  double t0, t1, t2, t_w;
  double dc_inv;
  double dlambda;
  double d_residual;
  double d_tmp;

  int ierr;
  int i, j;
  int n_iter;

  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  int n_local = N / numprocs;
  double *x = (double *)malloc(sizeof(double) * N);
  double *A_local = (double *)malloc(sizeof(double) * N * n_local);
  double *x_local = (double *)malloc(sizeof(double) * n_local);
  double *y_local = (double *)malloc(sizeof(double) * n_local);

  srand(1);
  dc_inv = 1.0 / (double)RAND_MAX;

  if (myid == 0) {
    for (j = 0; j < N; j++) {
      x[j] = rand() * dc_inv;
    }
    for (j = 0; j < N; j++) {
      for (i = j; i < N; i++) {
        A[j][i] = rand() * dc_inv;
        A[i][j] = A[j][i];
      }
    }
  }

  // printf("A matrix\n");
  // print_matrix(&A[0][0], N, N);

  MPI_Scatter(&A[0][0], N * n_local, MPI_DOUBLE, A_local, N * n_local,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(x, n_local, MPI_DOUBLE, x_local, n_local, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  /* Start of PowM routine ----------------------------*/
  ierr = MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  dlambda = PowM(A_local, x_local, y_local, N, n_local, &n_iter);

  //     ierr = MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  t0 = t2 - t1;
  ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  /* End of PowM routine ---------------------------- */

  /* Calculate residual */

  MPI_Gather(x_local, n_local, MPI_DOUBLE, x, n_local, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);
  d_residual = 0.0;
  for (i = 0; i < N; i++) {
    d_tmp = 0.0;
    for (j = 0; j < N; j++) {
      d_tmp += A[i][j] * x[j];
    }
    d_residual += (d_tmp - dlambda * x[i]) * (d_tmp - dlambda * x[i]);
  }
  d_residual = sqrt(d_residual);

  if (myid == 0) {
    printf("Nprocs = %d \n", numprocs);
    printf("N  = %d \n", N);
    printf("Power Method time  = %lf [sec.] \n", t_w);

    printf("Eigenvalue  = %e \n", dlambda);
    if (n_iter == -1) {
      printf("Not converged. \n");
    } else {
      printf("Iteration Number: %d \n", n_iter);
    }
    printf("Residual 2-Norm ||A x - lambda x||_2  = %e \n", d_residual);
  }

  ierr = MPI_Finalize();

  exit(0);
}

void MyMatVec(double *y_local, double *A_local, double *x, int n, int n_local) {
  int i, j;

  for (i = 0; i < n_local; i++) {
    y_local[i] = 0.0;
    for (j = 0; j < n; j++) {
      y_local[i] += A_local[i * n + j] * x[j];
    }
  }
}

double PowM(double *A_local, double *x_local, double *y_local, int n,
            int n_local, int *n_iter) {
  double d_tmp1, d_tmp2;
  double dlambda;
  double d_before = 0.0;

  int i;
  int i_loop;

  double *x = (double *)malloc(sizeof(double) * n);

  /* Normizeation of x */
  d_tmp1 = 0.0;
  for (i = 0; i < n_local; i++) {
    d_tmp1 += x_local[i] * x_local[i];
  }
  MPI_Allreduce(&d_tmp1, &d_tmp1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  d_tmp1 = 1.0 / sqrt(d_tmp1);
  for (i = 0; i < n_local; i++) {
    x_local[i] = x_local[i] * d_tmp1;
  }

  /* Main iteration loop ---------------------- */
  for (i_loop = 1; i_loop <= MAX_ITER; i_loop++) {
    if (DEBUG && myid == 0)
      printf("d_before = %e, dlambda = %e, EPS = %e \n", d_before, dlambda,
             EPS);

    /* Matrix Vector Product */
    MPI_Allgather(x_local, n_local, MPI_DOUBLE, x, n_local, MPI_DOUBLE,
                  MPI_COMM_WORLD);
    if (DEBUG && myid == 0)
      printf("x[0] = %e, x[1] = %e, x[2] = %e \n", x[0], x[1], x[2]);

    MyMatVec(y_local, A_local, x, n, n_local);

    /* innner products */
    d_tmp1 = 0.0;
    d_tmp2 = 0.0;
    for (i = 0; i < n_local; i++) {
      d_tmp1 += y_local[i] * y_local[i];
      d_tmp2 += y_local[i] * x_local[i];
    }

    MPI_Allreduce(&d_tmp1, &d_tmp1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&d_tmp2, &d_tmp2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* current approximately eigenvalue */
    dlambda = d_tmp1 / d_tmp2;

    /* Convergence test*/
    if (fabs(d_before - dlambda) < EPS) {
      *n_iter = i_loop;
      free(x);
      return dlambda;
    }

    /* keep current value */
    d_before = dlambda;

    /* Normalization and set new x */
    d_tmp1 = 1.0 / sqrt(d_tmp1);
    for (i = 0; i < n_local; i++)
      x_local[i] = y_local[i] * d_tmp1;

  } /* end of i_loop -------------------------- */

  free(x);
  *n_iter = -1;
  return dlambda;
}
