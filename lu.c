#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define N 128

#define MATRIX 1

#define EPS 2.220446e-16

int myid, numprocs;

void MyLUsolve(double A[N][N], double b[N], double x[N], int n);

int main(int argc, char *argv[]) {

  double t0, t1, t2, t_w;
  double dc_inv, d_mflops, dtemp, dtemp2, dtemp_t;

  int ierr;
  int i, j;
  int ii;
  int ib;

  ierr = MPI_Init(&argc, &argv);
  if (ierr != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, ierr);
    exit(1);
  }
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (ierr != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, ierr);
    exit(1);
  }
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  if (ierr != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, ierr);
    exit(1);
  }

  ib = N / numprocs;
  double *x = (double *)malloc(N * sizeof(double));
  double *b = (double *)malloc(N * sizeof(double));
  double *A = (double *)malloc(N * ib * sizeof(double));
  double Aall[N][N];

  /* matrix generation --------------------------*/
  if (MATRIX == 1) {
    for (j = 0; j < N; j++) {
      ii = 0;
      for (i = j; i < N; i++) {
        Aall[j][i] = (N - j) - ii;
        Aall[i][j] = Aall[j][i];
        ii++;
      }
    }

  } else {
    srand(1);
    dc_inv = 1.0 / (double)RAND_MAX;
    for (j = 0; j < N; j++) {
      for (i = 0; i < N; i++) {
        Aall[j][i] = rand() * dc_inv;
      }
    }
  } /* end of matrix generation -------------------------- */

  /* set matrix A  -------------------------- */

  for (i = 0; i < N; i++) {
    for (j = myid * ib; j < (myid + 1) * ib; j++) {
      A[i * ib + j - myid * ib] = Aall[i][j];
    }
  }

  /* set vector b  -------------------------- */
  for (i = 0; i < N; i++) {
    b[i] = 0.0;
    for (j = 0; j < N; j++) {
      b[i] += Aall[i][j];
    }
  }

  /* Start of LU routine ----------------------------*/
  ierr = MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  MyLUsolve(Aall, b, x, N);

  // ierr = MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  t0 = t2 - t1;
  ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  /* End of LU routine --------------------------- */

  if (myid == 0) {
    printf("N  = %d \n", N);
    printf("LU solve time  = %lf [sec.] \n", t_w);

    d_mflops = 2.0 / 3.0 * (double)N * (double)N * (double)N;
    d_mflops += 7.0 / 2.0 * (double)N * (double)N;
    d_mflops += 4.0 / 3.0 * (double)N;
    d_mflops = d_mflops / t_w;
    d_mflops = d_mflops * 1.0e-6;
    printf(" %lf [MFLOPS] \n", d_mflops);
  }

  /* Verification routine ----------------- */
  ib = N / numprocs;
  dtemp_t = 0.0;
  for (j = myid * ib; j < (myid + 1) * ib; j++) {
    dtemp2 = x[j] - 1.0;
    dtemp_t += dtemp2 * dtemp2;
  }
  dtemp_t = sqrt(dtemp_t);
  /* -------------------------------------- */

  MPI_Reduce(&dtemp_t, &dtemp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  /* Do not modify follows. -------- */
  if (myid == 0) {
    if (MATRIX == 1)
      dtemp2 = (double)N * (double)N * (double)N;
    else
      dtemp2 = (double)N * (double)N;
    dtemp_t = EPS * (double)N * dtemp2;
    printf("Pass value: %e \n", dtemp_t);
    printf("Calculated value: %e \n", dtemp);
    if (isnan(dtemp) || dtemp > dtemp_t) {
      printf("Error! Test is falled. \n");
      ierr = MPI_Finalize();
      exit(1);
    }
    printf(" OK! Test is passed. \n");
  }
  /* ----------------------------------------- */

  ierr = MPI_Finalize();

  exit(0);
}

void MyLUsolve(double A[N][N], double b[N], double *x, int n) {
  int ib = n / numprocs;
  int istart = myid * ib;
  int iend = (myid + 1) * ib;

  double *buf = (double *)malloc(n * sizeof(double));

  /* LU decomposition ---------------------- */
  for (int k = 0; k < iend; k++) {
    int k_proc = (k / ib);
    if (myid == k_proc) {
      double dtemp = 1.0 / A[k][k];
      for (int i = k + 1; i < n; i++) {
        A[i][k] = A[i][k] * dtemp;
        buf[i] = A[i][k];
      }
      for (int i = myid + 1; i < numprocs; i++) {
        MPI_Send(&buf[k + 1], n - k - 1, MPI_DOUBLE, i, k, MPI_COMM_WORLD);
      }
      istart = k + 1;
    } else {
      // } else if (myid > k_proc) {
      MPI_Recv(&buf[k + 1], n - k - 1, MPI_DOUBLE, k_proc, k, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    for (int col = k + 1; col < n; col++) {
      double dtemp = buf[col];
      for (int row = istart; row < iend; row++) {
        A[col][row] = A[col][row] - A[k][row] * dtemp;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* --------------------------------------- */

  istart = myid * ib;
  iend = (myid + 1) * ib;

  /* Forward substitution ------------------ */
  double *c = (double *)calloc(n, sizeof(double));

  for (int block = istart; block < n; block += ib) {
    if (myid != 0) {
      // from left
      MPI_Recv(&c[block], ib, MPI_DOUBLE, myid - 1, block, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
    if (block / ib == myid) {
      for (int i = block; i < block + ib; i++) {
        c[i] = b[i] + c[i];
        int tmpend = istart + i - block;
        for (int j = istart; j < tmpend; j++) {
          c[i] -= A[i][j] * c[j];
        }
      }
    } else {
      for (int i = block; i < block + ib; i++) {
        for (int j = istart; j < iend; j++) {
          c[i] -= A[i][j] * c[j];
        }
      }
      if (myid != numprocs - 1) {
        // to right
        MPI_Send(&c[block], ib, MPI_DOUBLE, myid + 1, block, MPI_COMM_WORLD);
      }
    }
  }
  // for debug
  // MPI_Gather(&c[istart], ib, MPI_DOUBLE, &c[0], ib, MPI_DOUBLE, numprocs - 1,
  //            MPI_COMM_WORLD);
  // if (myid == numprocs - 1) {
  //   for (int i = 0; i < n; i++) {
  //     printf("%lf ", c[i]);
  //   }
  //   printf("\n");
  // }
  MPI_Barrier(MPI_COMM_WORLD);
  /* --------------------------------------- */

  /* Backward substitution ------------------ */

  for (int block = istart; block >= 0; block -= ib) {
    if (myid != numprocs - 1) {
      // from right
      MPI_Recv(&x[block], ib, MPI_DOUBLE, myid + 1, block, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
    if (block / ib == myid) {
      for (int i = iend - 1; i >= istart; i--) {
        x[i] += c[i];
        for (int j = iend - 1; j > i; j--) {
          x[i] -= A[i][j] * x[j];
        }
        x[i] = x[i] / A[i][i];
      }
    } else {
      for (int i = block + ib - 1; i >= block; i--) {
        for (int j = istart; j < iend; j++) {
          x[i] -= A[i][j] * x[j];
        }
      }
      if (myid != 0) {
        // to left
        MPI_Send(&x[block], ib, MPI_DOUBLE, myid - 1, block, MPI_COMM_WORLD);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Gather(&x[istart], ib, MPI_DOUBLE, &x[0], ib, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);
  /* --------------------------------------- */
}
