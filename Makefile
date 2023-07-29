# ----------------------------------------------------------------
# environment
CC		= mpifccpx
FC		= 

# ----------------------------------------------------------------
# options

CFLAGS          = -Kfast -Koptmsg=2
FFLAGS		= 

# ----------------------------------------------------------------
# sources and objects

C_SRC		= $(wildcard *.c)
BASH_SRC	= $(C_SRC:.c=.bash)
F_SRC		= 

C_OBJ		= $(C_SRC:.c=)
F_OBJ		= $(F_SRC:.f=)

# ----------------------------------------------------------------
# executables

EXEC		= $(C_OBJ) 
ARG = transpose

all:		$(EXEC) $(BASH_SRC)

run_all:	$(EXEC) $(BASH_SRC)
	for i in $(EXEC); do \
		pjsub $$i.bash; \
	done

run:		$(ARG) $(ARG).bash
	pjsub $(ARG).bash

# $(C_OBJ):	$(C_SRC)
# 	$(CC) -o $@ $(CFLAGS) $(C_SRC) -lm


# ----------------------------------------------------------------
# rules

# from .c to executable
%:	%.c
	$(CC) -o $@ $(CFLAGS) $< -lm

%.bash:	
	./make_bash.bash $(@:.bash=)

.f.:
	$(FC) -o $* $(FFLAGS) -c $<

# ----------------------------------------------------------------
# clean up

clean:
	/bin/rm -f $(EXEC) $(F_SRC:.f=.o) *.optrpt *.out *.err $(BASH_SRC)

.PHONY:		all clean run

# ----------------------------------------------------------------
# End of Makefile
