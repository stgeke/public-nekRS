#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include <unistd.h>
#include "mpi.h"
#include <vector>
#include <algorithm>

#include "nrssys.hpp"
#include "setupAide.hpp"
#include "platform.hpp"
#include "configReader.hpp"

namespace {

size_t wordSize = 8;

occa::kernel fdmKernel;
occa::kernel oldFdmKernel;

occa::memory o_Sx;
occa::memory o_Sy;
occa::memory o_Sz;
occa::memory o_invL;
occa::memory o_u;
occa::memory o_Su;
occa::memory o_SuGold;

int Np; 
int Nelements; 

template<typename FloatingPointType>
double checkCorrectnessImpl(occa::memory & o_a, occa::memory & o_b){
  FloatingPointType linfError = 0.0;

  std::vector<FloatingPointType> results_a(Np * Nelements, 0.0);
  std::vector<FloatingPointType> results_b(Np * Nelements, 0.0);
  o_a.copyTo(results_a.data(), Np * Nelements * sizeof(FloatingPointType));
  o_b.copyTo(results_b.data(), Np * Nelements * sizeof(FloatingPointType));

  for(int i = 0; i < Np * Nelements; ++i){
    linfError = std::max(linfError, std::abs(results_a[i] - results_b[i]));
  }

  return static_cast<double>(linfError);
}

double checkCorrectness(occa::memory & o_a, occa::memory & o_b)
{
  double linfError = -100.0;
  if(wordSize == 4){
    linfError = checkCorrectnessImpl<float>(o_a, o_b);
  }
  
  if(wordSize == 8){
    linfError = checkCorrectnessImpl<double>(o_a, o_b);
  }

  MPI_Allreduce(MPI_IN_PLACE, &linfError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return linfError;
}

std::pair<double, double> run(int Ntests, bool performCorrectnessCheck = false)
{

  double error = -100.0;
  
  if(performCorrectnessCheck){
    // correctness check
    oldFdmKernel(Nelements, o_Su, o_Sx, o_Sy, o_Sz, o_invL, o_u);
    fdmKernel(Nelements, o_SuGold, o_Sx, o_Sy, o_Sz, o_invL, o_u);

    error = checkCorrectness(o_Su, o_SuGold);
  }

  platform->device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();

  for(int test = 0; test < Ntests; ++test) {
    fdmKernel(Nelements, o_Su, o_Sx, o_Sy, o_Sz, o_invL, o_u);
  }

  platform->device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  return std::make_pair((MPI_Wtime() - start) / Ntests, error);
} 

void* (*randAlloc)(int);

void* rand32Alloc(int N)
{
  float* v = (float*) malloc(N * sizeof(float));

  for(int n = 0; n < N; ++n)
    v[n] = drand48();

  return v;
}

void* rand64Alloc(int N)
{
  double* v = (double*) malloc(N * sizeof(double));

  for(int n = 0; n < N; ++n)
    v[n] = drand48();

  return v;
}

} // namespace

int main(int argc, char** argv)
{
  int rank = 0, size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  configRead(MPI_COMM_WORLD);
  std::string installDir(getenv("NEKRS_HOME"));
  setupAide options; 

  int err = 0;
  int cmdCheck = 0;

  int N;
  int okl = 1;
  int Ntests = -1;

  while(1) {
    static struct option long_options[] =
    {
      {"p-order", required_argument, 0, 'p'},
      {"elements", required_argument, 0, 'e'},
      {"backend", required_argument, 0, 'b'},
      {"arch", required_argument, 0, 'a'},
      {"fp32", no_argument, 0, 'f'},
      {"help", required_argument, 0, 'h'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}
    };
    int option_index = 0;
    int c = getopt_long (argc, argv, "", long_options, &option_index);

    if (c == -1)
      break;

    switch(c) {
    case 'p':
      N = atoi(optarg); 
      cmdCheck++; 
      break;
    case 'e':
      Nelements = atoi(optarg);
      cmdCheck++;
      break;
    case 'b':
      options.setArgs("THREAD MODEL", std::string(optarg));
      cmdCheck++;
      break;
    case 'f':
      wordSize = 4;;
      break;
    case 'i':
      Ntests = atoi(optarg);
      break;
    case 'h':
      err = 1;
      break;
    default:
      err = 1;
    }
  }

  if(err || cmdCheck != 3) {
    if(rank == 0)
      printf("Usage: ./nekrs-fdm  --p-order <n> --elements <n> --backend <CPU|CUDA|HIP|OPENCL>\n"
             "                    [--fp32] [--iterations <n>]\n"); 
    exit(1); 
  }

  if(N <= 2){
    if(rank == 0){
      printf("Error: N > 2!\n");
    }
    exit(1);
  }

  Nelements = std::max(1, Nelements/size);
  const int Nq = N + 1;
  const int Np = Nq * Nq * Nq;

  platform = platform_t::getInstance(options, MPI_COMM_WORLD, MPI_COMM_WORLD); 
  const int Nthreads =  omp_get_max_threads();

  // build+load kernel
  occa::properties props = platform->kernelInfo + meshKernelProperties(N-2); // regular, non-extended mesh
  if(wordSize == 4) props["defines/pfloat"] = "float";
  else props["defines/pfloat"] = "dfloat";

  props["defines/p_Nq_e"] = Nq;
  props["defines/p_Np_e"] = Np;
  props["defines/p_overlap"] = 0;

  // always benchmark ASM
  props["defines/p_restrict"] = 0;
  props["defines/p_knl"] = 0;

  auto oldKernelProps = props;
  oldKernelProps["defines/p_knl"] = -1;

  std::string kernelName = "fusedFDM";
  const std::string ext = (platform->device.mode() == "Serial") ? ".c" : ".okl";
  std::string fileName = installDir + "/okl/elliptic/" + kernelName + ext;
  oldFdmKernel = platform->device.buildKernel(fileName, oldKernelProps, true);

  // populate arrays
  randAlloc = &rand64Alloc; 
  if(wordSize == 4) randAlloc = &rand32Alloc;

  void *Sx   = randAlloc(Nelements * Nq * Nq);
  void *Sy   = randAlloc(Nelements * Nq * Nq);
  void *Sz   = randAlloc(Nelements * Nq * Nq);
  void *invL = randAlloc(Nelements * Np);
  void *Su   = randAlloc(Nelements * Np);
  void *u    = randAlloc(Nelements * Np);

  o_Sx = platform->device.malloc(Nelements * Nq * Nq * wordSize, Sx);
  free(Sx);
  o_Sy = platform->device.malloc(Nelements * Nq * Nq * wordSize, Sy);
  free(Sy);
  o_Sz = platform->device.malloc(Nelements * Nq * Nq * wordSize, Sz);
  free(Sz);
  o_invL = platform->device.malloc(Nelements * Np * wordSize, invL);
  free(invL);
  o_Su = platform->device.malloc(Nelements * Np * wordSize, Su);
  o_SuGold = platform->device.malloc(Nelements * Np * wordSize, Su);
  free(Su);
  o_u = platform->device.malloc(Nelements * Np * wordSize, u);
  free(u);

  constexpr int Nkernels = 11;

  // v8 is only valid for even p_Nq_e

  for(int knl = 0; knl <= Nkernels; ++knl){
    if(knl == 8 && Nq % 2 == 1) continue;
    auto newProps = props;
    newProps["defines/p_knl"] = knl;

    if(platform->device.mode() == "HIP"){
      props["defines/OCCA_USE_HIP"] = 1;
    }

    kernelName = "fusedFDM";
    fileName = 
      installDir + "/okl/elliptic/" + kernelName + ext;

    fdmKernel = platform->device.buildKernel(fileName, newProps, true);

    // warm-up
    auto elapsedAndError = run(10, true);
    auto elapsed = elapsedAndError.first;
    auto error = elapsedAndError.second;
    const int elapsedTarget = 10;
    if(Ntests < 0) Ntests = elapsedTarget/elapsed;

    // ***** 
    elapsedAndError = run(Ntests, false);
    // ***** 

    elapsed = elapsedAndError.first;
 
    // print statistics
    const dfloat GDOFPerSecond = (size * Nelements * (N* N * N) / elapsed) / 1.e9;

    size_t bytesPerElem = (3 * Np + 3 * Nq * Nq) * wordSize;
    const double bw = (size * Nelements * bytesPerElem / elapsed) / 1.e9;

    double flopsPerElem = 12 * Nq * Np + Np;
    const double gflops = (size * flopsPerElem * Nelements / elapsed) / 1.e9;

    if(rank == 0)
      std::cout << "MPItasks=" << size
                << " OMPthreads=" << Nthreads
                << " NRepetitions=" << Ntests
                << " N=" << N
                << " Nelements=" << size * Nelements
                << " error=" << error
                << " elapsed time=" << elapsed
                << " wordSize=" << 8*wordSize
                << " GDOF/s=" << GDOFPerSecond
                << " GB/s=" << bw
                << " GFLOPS/s=" << gflops
                << " kernel=" << knl
                << "\n";
  }

  MPI_Finalize();
  exit(0);
}
