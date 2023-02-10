#if !defined(nekrs_nrssys_hpp_)
#define nekrs_nrssys_hpp_

#define BLOCKSIZE 256
#define ALIGN_SIZE 4096

//float data type
#if 0
using dfloat = float;
#define DFLOAT_SINGLE
#define MPI_DFLOAT MPI_FLOAT
#define dfloatFormat "%f"
#define dfloatString "float"
#else
using dfloat = double;
#define DFLOAT_DOUBLE
#define MPI_DFLOAT MPI_DOUBLE
#define dfloatFormat "%lf"
#define dfloatString "double"
#endif

//smoother float data type
#if 1
using pfloat = float;
#define MPI_PFLOAT MPI_FLOAT
#define pfloatFormat "%f"
#define pfloatString "float"
#else
using pfloat = double;
#define MPI_PFLOAT MPI_DOUBLE
#define pfloatFormat "%lf"
#define pfloatString "double"
#endif

//host index data type
#if 0
using hlong = int;
#define MPI_HLONG MPI_INT
#define hlongFormat "%d"
#define hlongString "int"
#else
using hlong = long long int;
#define MPI_HLONG MPI_LONG_LONG_INT
#define hlongFormat "%lld"
#define hlongString "long long int"
#endif

//device index data type
#if 1
using dlong = int;
#define MPI_DLONG MPI_INT
#define dlongFormat "%d"
#define dlongString "int"
#else
using dlong = long long int;
#define MPI_DLONG MPI_LONG_LONG_INT;
#define dlongFormat "%lld"
#define dlongString "long long int"
#endif

// Workaround for https://github.com/open-mpi/ompi/issues/5157
#define OMPI_SKIP_MPICXX 1

#include <mpi.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <getopt.h>
#include <sys/stat.h>

#include "occa.hpp"
#include "ogs.hpp"
#include "setupAide.hpp"
#include "timer.hpp"

#define nrsCheck(cond, comm, exitCode, message, ...) \
  do { \
    int _nrsCheckErr = 0; \
    if(cond) _nrsCheckErr = 1; \
    MPI_Allreduce(MPI_IN_PLACE, &_nrsCheckErr, 1, MPI_INT, MPI_SUM, comm); \
    if(_nrsCheckErr) { \
      int rank = 0; \
      MPI_Comm_rank(comm, &rank); \
      if(rank == 0) { \
        printf("Error in %s: ", __func__);\
        printf(message, __VA_ARGS__); \
      } \
      fflush(stdout); \
      MPI_Barrier(comm); \
      MPI_Abort(MPI_COMM_WORLD, exitCode); \
    } \
  } while (0)

#define nrsAbort(...) \
  do { \
    nrsCheck(true, __VA_ARGS__); \
  } while (0)

static occa::memory o_NULL;

struct platform_t;
extern platform_t* platform;

#define NSCALAR_MAX 100
static const std::string scalarDigitStr(int i)
{
  static const int scalarWidth = std::to_string(NSCALAR_MAX - 1).length();
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(scalarWidth) << i;
  return ss.str();
};

#define EXIT_AND_FINALIZE(a)  { fflush(stdout); MPI_Finalize(); exit(a); }
#define ABORT(a)  { fflush(stdout); MPI_Abort(MPI_COMM_WORLD, a); }

#endif
