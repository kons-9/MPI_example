#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define N 1152
#define DEBUG 1
#define EPS 1.0e-18

static double A[N][N];
static double B[N][N];
static double C[N][N];

int myid, numprocs;

void get_topology_info(int root_numprocs, int *left, int *right, int *up,
                       int *down, MPI_Comm *comm) {
  int dim[2] = {root_numprocs, root_numprocs};
  int periods[2] = {1, 1};
  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periods, 1, comm);
  MPI_Cart_shift(*comm, 0, 1, left, right);
  MPI_Cart_shift(*comm, 1, 1, up, down);
}

void calculate_local_matmal(double *A, double *B, double *C, int Nl) {
  int i, j, k;
  for (i = 0; i < Nl; i++)
    for (k = 0; k < Nl; k++)
      for (j = 0; j < Nl; j++)
        C[i * Nl + j] += A[i * Nl + k] * B[k * Nl + j];
}

void solve(double *A, double *B, double *C, int Nl, int root_numproc, int left,
           int right, int up, int down, MPI_Comm cannon_comm) {

  double *ALocal = (double *)malloc(Nl * Nl * sizeof(double));
  double *BLocal = (double *)malloc(Nl * Nl * sizeof(double));
  double *CLocal = (double *)malloc(Nl * Nl * sizeof(double));

  MPI_Scatter(A, Nl * Nl, MPI_DOUBLE, ALocal, Nl * Nl, MPI_DOUBLE, 0,
              cannon_comm);
  MPI_Scatter(B, Nl * Nl, MPI_DOUBLE, BLocal, Nl * Nl, MPI_DOUBLE, 0,
              cannon_comm);

  int num_shift;
  double *buf = (double *)malloc(Nl * Nl * sizeof(double));
  double *tmp;

  for (num_shift = 0; num_shift < root_numproc - 1; num_shift++) {
    // Matrix multiplication
    calculate_local_matmal(ALocal, BLocal, CLocal, Nl);
    // Communication
    MPI_Sendrecv_replace(ALocal, Nl * Nl, MPI_DOUBLE, left, 1, right, 1,
                         cannon_comm, MPI_STATUS_IGNORE);
    // MPI_Sendrecv(ALocal, Nl * Nl, MPI_DOUBLE, left, 1, buf, Nl * Nl,
    // MPI_DOUBLE,
    //              right, 1, cannon_comm, MPI_STATUS_IGNORE);
    // tmp = buf;
    // buf = ALocal;
    // ALocal = tmp;
    MPI_Sendrecv_replace(BLocal, Nl * Nl, MPI_DOUBLE, up, 2, down, 2,
                         cannon_comm, MPI_STATUS_IGNORE);
    // MPI_Sendrecv(BLocal, Nl * Nl, MPI_DOUBLE, up, 2, buf, Nl * Nl,
    // MPI_DOUBLE,
    //              down, 2, cannon_comm, MPI_STATUS_IGNORE);
    // tmp = buf;
    // buf = BLocal;
    // BLocal = tmp;
  }
  calculate_local_matmal(ALocal, BLocal, CLocal, Nl);

  MPI_Gather(CLocal, Nl * Nl, MPI_DOUBLE, C, Nl * Nl, MPI_DOUBLE, 0, cannon_comm);
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
  int left, right, up, down;
  MPI_Comm cannon_comm;
  get_topology_info(root_numprocs, &left, &right, &up, &down, &cannon_comm);
  int Nl = N / root_numprocs;

  if (myid == 0) {
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
      initialize_matrix(A, N);
    }
  }

  ierr = MPI_Barrier(cannon_comm);
  t1 = MPI_Wtime();

  solve(A, B, C, Nl, root_numprocs, left, right, up, down, cannon_comm);

  t2 = MPI_Wtime();
  t0 = t2 - t1;
  ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, cannon_comm);

  if (myid == 0) {
    printf("N  = %d \n", N);
    printf("Mat-Mat time  = %lf [sec.] \n", t_w);

    double d_mflops = 2.0 * (double)N * (double)N * (double)N / t_w;
    d_mflops = d_mflops * 1.0e-6;
    printf(" %lf [MFLOPS] \n", d_mflops);
  }

  rc = MPI_Finalize();
  exit(0);
}
