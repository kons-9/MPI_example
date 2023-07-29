# !/bin/bash
#PJM -L rscgrp=lecture-o
#PJM -L node=12
#PJM --mpi proc=576
#PJM -L elapse=00:01:00
#PJM -g gt13

module load odyssey

mpiexec ./mat_vec
