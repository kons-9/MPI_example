#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define N 2400
#define DEBUG 1
#define EPS 1.0e-18

static double A[N][N];
static double B[N][N];
static double C[N][N];

int myid, numprocs;

void get_topology_info(int root_numprocs, MPI_Comm *comm) {
  int dim[2] = {root_numprocs, root_numprocs};
  int periods[2] = {1, 1};
  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periods, 1, comm);
}

void calculate_local_matmal(double *A, double *B, double *C, int Nl) {
  int i, j, k;
  int tmpa;
  for (i = 0; i < Nl; i++) {
    for (k = 0; k < Nl; k++) {
      tmpa = A[i * Nl + k];
      for (j = 0; j < Nl; j++) {
        C[i * Nl + j] += tmpa * B[k * Nl + j];
      }
    }
  }
}

void solve(double A[N][N], double B[N][N], double C[N][N], int Nl,
           int root_numproc, MPI_Comm cannon_comm) {

  int i, j;

  double *localA = (double *)malloc(Nl * Nl * sizeof(double));
  double *localB = (double *)malloc(Nl * Nl * sizeof(double));
  double *localC = (double *)calloc(Nl * Nl, sizeof(double));

  MPI_Datatype my_mat_type, type;
  MPI_Type_vector(Nl, Nl, N, MPI_DOUBLE, &type);
  MPI_Type_create_resized(type, 0, Nl * sizeof(double), &my_mat_type);
  MPI_Type_commit(&my_mat_type);

  int *sendcounts = (int *)malloc(numprocs * sizeof(int));
  int *displs = (int *)malloc(numprocs * sizeof(int));

  int disp = 0;
  for (i = 0; i < root_numproc; i++) {
    for (j = 0; j < root_numproc; j++) {
      sendcounts[i * root_numproc + j] = 1;
      displs[i * root_numproc + j] = disp;
      disp += 1;
    }
    disp += (Nl - 1) * root_numproc;
  }

  MPI_Scatterv(&A[0][0], sendcounts, displs, my_mat_type, localA, Nl * Nl,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatterv(&B[0][0], sendcounts, displs, my_mat_type, localB, Nl * Nl,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // skewing
  int coord[2];
  int left, right, up, down;
  MPI_Cart_coords(cannon_comm, myid, 2, coord);
  MPI_Cart_shift(cannon_comm, 1, coord[0], &left, &right);
  MPI_Sendrecv_replace(localA, Nl * Nl, MPI_DOUBLE, left, 1, right, 1,
                       cannon_comm, MPI_STATUS_IGNORE);
  MPI_Cart_shift(cannon_comm, 0, coord[1], &up, &down);
  MPI_Sendrecv_replace(localB, Nl * Nl, MPI_DOUBLE, up, 1, down, 1, cannon_comm,
                       MPI_STATUS_IGNORE);

  int num_shift;

  MPI_Cart_shift(cannon_comm, 1, 1, &left, &right);
  MPI_Cart_shift(cannon_comm, 0, 1, &up, &down);
  for (num_shift = 0; num_shift < root_numproc - 1; num_shift++) {
    // Matrix multiplication
    calculate_local_matmal(localA, localB, localC, Nl);

    MPI_Sendrecv_replace(localA, Nl * Nl, MPI_DOUBLE, left, 1, right, 1,
                         cannon_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(localB, Nl * Nl, MPI_DOUBLE, up, 2, down, 2,
                         cannon_comm, MPI_STATUS_IGNORE);
  }
  calculate_local_matmal(localA, localB, localC, Nl);

  MPI_Gatherv(localC, Nl * Nl, MPI_DOUBLE, &C[0][0], sendcounts, displs,
              my_mat_type, 0, MPI_COMM_WORLD);
  return;
}

void initialize_matrix(double A[N][N], int n) {
  int i, j;
  double dc_inv = 1.0 / (double)RAND_MAX;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      A[i][j] = rand() * dc_inv;
    }
  }
}

int main(int argc, char *argv[]) {

  int ierr, rc;
  int i, j;
  double dc_inv;
  double t0, t1, t2, t_w;

  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  int root_numprocs = sqrt(numprocs);
  MPI_Comm cannon_comm;
  get_topology_info(root_numprocs, &cannon_comm);
  int Nl = N / root_numprocs;

  if (myid == 0) {
    if (DEBUG == 1) {
      srand(1);
      dc_inv = 1.0 / (double)RAND_MAX;
      for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
          A[i][j] = 1.0;
          B[i][i] = 1.0;
        }
      }
    } else {
      srand(1);
      initialize_matrix(A, N);
      initialize_matrix(B, N);
    }
  }

  ierr = MPI_Barrier(cannon_comm);
  t1 = MPI_Wtime();

  solve(A, B, C, Nl, root_numprocs, cannon_comm);

  t2 = MPI_Wtime();
  t0 = t2 - t1;
  ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, cannon_comm);

  if (myid == 0) {
    printf("N  = %d \n", N);
    printf("Mat-Mat time  = %lf [sec.] \n", t_w);

    double d_mflops = 2.0 * (double)N * (double)N * (double)N / t_w;
    d_mflops = d_mflops * 1.0e-6;
    printf(" %lf [MFLOPS] \n", d_mflops);
    if (DEBUG) {
      for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
          if (C[i][j] != 1) {
            printf("Error! C[%d][%d]=%lf\n", i, j, C[i][j]);
            rc = MPI_Finalize();
            exit(1);
          }
        }
      }
      printf("OK!\n");
    }
  }

  rc = MPI_Finalize();
  exit(0);
}
