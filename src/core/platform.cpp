#include <cstdlib>
#include <strings.h>

#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "flopCounter.hpp"
#include "fileUtils.hpp"

namespace {

static void compileDummyKernel(platform_t &plat)
{
  const bool buildNodeLocal = plat.cacheLocal;
  auto rank = buildNodeLocal ? plat.comm.localRank : plat.comm.mpiRank;
  const std::string dummyKernelName = "myDummyKernelName";
  const std::string dummyKernelStr = std::string("@kernel void myDummyKernelName(int N) {"
                                                 "  for (int i = 0; i < N; ++i; @tile(64, @outer, @inner)) {}"
                                                 "}");

  if (rank == 0) {
    plat.device.occaDevice().buildKernelFromString(dummyKernelStr, dummyKernelName, plat.kernelInfo);
  }
}

} // namespace

deviceVector_t::deviceVector_t(const size_t _offset,
                               const size_t _nVectors,
                               const size_t _wordSize,
                               const std::string _vectorName)
    : nVectors(_nVectors), wordSize(_wordSize), vectorName(_vectorName), offset(_offset)
{
  nrsCheck(offset <= 0 || nVectors <= 0 || wordSize <= 0, MPI_COMM_SELF, EXIT_FAILURE,
           "deviceVector_t invalid input!\n", "");

  o_vector = platform->device.malloc(nVectors * offset * wordSize);
  for (int s = 0; s < nVectors; ++s) {
    slices.push_back(o_vector + (s * wordSize) * offset);
  }
}

occa::memory &deviceVector_t::at(const int i)
{
  nrsCheck(i >= nVectors, MPI_COMM_SELF, EXIT_FAILURE,
           "deviceVector_t(%s) has %zu size, but an attempt to access entry %i was made!\n",
           vectorName.c_str(), nVectors, i);

  return slices[i];
}

platform_t *platform_t::singleton = nullptr;
platform_t::platform_t(setupAide &_options, MPI_Comm _commg, MPI_Comm _comm)
    : options(_options), warpSize(32), comm(_commg, _comm), device(options, comm),
      timer(_comm, device.occaDevice(), 0, 0), kernels(*this)
{
  int rank;
  MPI_Comm_rank(_comm, &rank);

  exitValue = 0;

  setenv("OCCA_MEM_BYTE_ALIGN", "1024", 1);

  cacheLocal = 0;
  if(getenv("NEKRS_CACHE_LOCAL"))
    cacheLocal = std::stoi(getenv("NEKRS_CACHE_LOCAL"));

  cacheBcast = 0;
  if(getenv("NEKRS_CACHE_BCAST"))
    cacheBcast = std::stoi(getenv("NEKRS_CACHE_BCAST"));

  nrsCheck(cacheLocal && cacheBcast,
           _comm, EXIT_FAILURE, 
           "NEKRS_CACHE_LOCAL=1 and NEKRS_CACHE_BCAST=1 is incompatible!", "");

  srand48((long int)comm.mpiRank);

  oogs::gpu_mpi(std::stoi(getenv("NEKRS_GPU_MPI")));

  verbose = options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;

  timer.enableSync();
  if (options.compareArgs("ENABLE TIMER SYNC", "FALSE"))
    timer.disableSync();

  flopCounter = std::make_unique<flopCounter_t>();

  {
    int N;
    options.getArgs("POLYNOMIAL DEGREE", N);
    const int Nq = N + 1;
    nrsCheck(BLOCKSIZE < Nq * Nq, comm.mpiComm, EXIT_FAILURE,
             "Some kernels require BLOCKSIZE >= Nq * Nq\nBLOCKSIZE = %d, Nq*Nq = %d\n",
             BLOCKSIZE, Nq * Nq);
  }

  // create tmp dir
  {
    int rankLocal;
    MPI_Comm_rank(comm.mpiCommLocal, &rankLocal);

    char tmp[] = "nrs_XXXXXX";
    const int tmpSize = sizeof(tmp)/sizeof(tmp[0]);
    
    int retVal = 0;
    if(rankLocal == 0) { 
      retVal = mkstemp(tmp);
      fs::remove(fs::path(tmp));
    }
   
    MPI_Bcast(&tmp, tmpSize, MPI_CHAR, 0, comm.mpiComm);
    if(getenv("NEKRS_TMP_DIR")) 
      tmpDir = getenv("NEKRS_TMP_DIR");
    else
      tmpDir = fs::temp_directory_path() / fs::path(tmp);

    if(rankLocal == 0) { 
      fs::create_directory(tmpDir);
      nrsCheck(!fs::exists(tmpDir), MPI_COMM_SELF, EXIT_FAILURE,
               "Cannot create %s\n", tmpDir.c_str());
    }
  }

  // bcast install dir 
  if(cacheBcast || cacheLocal) {
    const auto NEKRS_HOME_NEW = fs::path(tmpDir) / "nekrs";
    const auto srcPath = fs::path(getenv("NEKRS_HOME"));
    for (auto &entry : {fs::path("udf"),
                        fs::path("nek5000"),
                        fs::path("nekInterface"),
                        fs::path("include"),
                        fs::path("gatherScatter"),
                        fs::path("kernels")}) {
      fileBcast(srcPath / entry, NEKRS_HOME_NEW, comm.mpiComm, verbose);
    }
    setenv("NEKRS_HOME", std::string(NEKRS_HOME_NEW).c_str(), 1);
    setenv("NEKRS_KERNEL_DIR", std::string(NEKRS_HOME_NEW / "kernels").c_str(), 1);
    setenv("OGS_HOME", std::string(NEKRS_HOME_NEW / "gatherScatter").c_str(), 1);
  }

  {
    int rankLocal = rank;
    if(cacheLocal)
      MPI_Comm_rank(comm.mpiCommLocal, &rankLocal);
 
    if(rankLocal == 0) {
      std::string cache_dir;
      cache_dir.assign(getenv("NEKRS_CACHE_DIR"));
      mkdir(cache_dir.c_str(), S_IRWXU);
    }
  }

  // Disables the automatic insertion of barriers
  // between separate OKL inner loop blocks.
  kernelInfo["okl/add_barriers"] = false;

  kernelInfo["defines/"
             "p_NVec"] = 3;
  kernelInfo["defines/"
             "p_blockSize"] = BLOCKSIZE;
  kernelInfo["defines/"
             "dfloat"] = dfloatString;
  kernelInfo["defines/"
             "pfloat"] = pfloatString;
  kernelInfo["defines/"
             "dlong"] = dlongString;
  kernelInfo["defines/"
             "hlong"] = hlongString;

  if (device.mode() == "CUDA") {
  }

  if (device.mode() == "OpenCL") {
    kernelInfo["defines/"
               "hlong"] = "long";
  }

  if (device.mode() == "HIP") {
    warpSize = 64; // can be arch specific
  }

  serial = device.mode() == "Serial" || device.mode() == "OpenMP";

  const std::string extension = serial ? ".c" : ".okl";

  compileDummyKernel(*this);

  std::string kernelName, fileName;
  const auto oklpath = std::string(getenv("NEKRS_KERNEL_DIR"));
  kernelName = "copyDfloatToPfloat";
  fileName = oklpath + "/core/" + kernelName + extension;
  this->kernels.add(kernelName, fileName, this->kernelInfo);

  kernelName = "copyPfloatToDfloat";
  fileName = oklpath + "/core/" + kernelName + extension;
  this->kernels.add(kernelName, fileName, this->kernelInfo);
}
void memPool_t::allocate(const dlong offset, const dlong fields)
{
  if (ptr)
    free(ptr);

  ptr = (dfloat *)calloc(offset * fields, sizeof(dfloat));
  if (fields > 0)
    slice0 = ptr + 0 * offset;
  if (fields > 1)
    slice1 = ptr + 1 * offset;
  if (fields > 2)
    slice2 = ptr + 2 * offset;
  if (fields > 3)
    slice3 = ptr + 3 * offset;
  if (fields > 4)
    slice4 = ptr + 4 * offset;
  if (fields > 5)
    slice5 = ptr + 5 * offset;
  if (fields > 6)
    slice6 = ptr + 6 * offset;
  if (fields > 7)
    slice7 = ptr + 7 * offset;
  if (fields > 9)
    slice9 = ptr + 9 * offset;
  if (fields > 12)
    slice12 = ptr + 12 * offset;
  if (fields > 15)
    slice15 = ptr + 15 * offset;
  if (fields > 18)
    slice18 = ptr + 18 * offset;
  if (fields > 19)
    slice19 = ptr + 19 * offset;
}
void deviceMemPool_t::allocate(memPool_t &hostMemory, const dlong offset, const dlong fields)
{
  if (o_ptr.size())
    o_ptr.free();

  bytesAllocated = (fields * sizeof(dfloat)) * offset;
  o_ptr = platform->device.malloc(bytesAllocated, hostMemory.slice0);
  if (fields > 0)
    slice0 = o_ptr.slice((0 * sizeof(dfloat)) * offset);
  if (fields > 1)
    slice1 = o_ptr.slice((1 * sizeof(dfloat)) * offset);
  if (fields > 2)
    slice2 = o_ptr.slice((2 * sizeof(dfloat)) * offset);
  if (fields > 3)
    slice3 = o_ptr.slice((3 * sizeof(dfloat)) * offset);
  if (fields > 4)
    slice4 = o_ptr.slice((4 * sizeof(dfloat)) * offset);
  if (fields > 5)
    slice5 = o_ptr.slice((5 * sizeof(dfloat)) * offset);
  if (fields > 6)
    slice6 = o_ptr.slice((6 * sizeof(dfloat)) * offset);
  if (fields > 7)
    slice7 = o_ptr.slice((7 * sizeof(dfloat)) * offset);
  if (fields > 9)
    slice9 = o_ptr.slice((9 * sizeof(dfloat)) * offset);
  if (fields > 12)
    slice12 = o_ptr.slice((12 * sizeof(dfloat)) * offset);
  if (fields > 15)
    slice15 = o_ptr.slice((15 * sizeof(dfloat)) * offset);
  if (fields > 18)
    slice18 = o_ptr.slice((18 * sizeof(dfloat)) * offset);
  if (fields > 19)
    slice19 = o_ptr.slice((19 * sizeof(dfloat)) * offset);
}

void platform_t::create_mempool(const dlong offset, const dlong fields)
{
  mempool.allocate(offset, fields);
  o_mempool.allocate(mempool, offset, fields);
}
