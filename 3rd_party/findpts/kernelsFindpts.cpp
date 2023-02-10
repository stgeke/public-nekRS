// for platform
#include "nrssys.hpp"
#include "nrs.hpp"

#include "ogstypes.h"
#include "findpts.hpp"
#include <cfloat>
#include <tuple>
#include <limits>

namespace findpts {

// compute nearest power of two larger than v
// from: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
unsigned nearestPowerOfTwo(unsigned int v)
{
  static_assert(sizeof(unsigned int) == 4);
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

std::vector<occa::kernel> findpts_t::initFindptsKernels(dlong Nq)
{
  constexpr int dim = 3;

  occa::properties kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  kernelInfo["defines/p_D"] = dim;
  kernelInfo["defines/p_Nq"] = Nq;
  kernelInfo["defines/p_Np"] = Nq * Nq * Nq;
  kernelInfo["defines/p_nptsBlock"] = 4;

  unsigned int Nq2 = Nq * Nq;
  const auto blockSize = nearestPowerOfTwo(Nq2);

  const auto oklpath = std::string(getenv("FINDPTS_HOME")) + "/okl/";

  kernelInfo["defines/p_blockSize"] = blockSize;
  kernelInfo["defines/p_Nfp"] = Nq * Nq;
  kernelInfo["defines/dlong"] = dlongString;
  kernelInfo["defines/hlong"] = hlongString;
  kernelInfo["defines/dfloat"] = dfloatString;
  kernelInfo["defines/DBL_MAX"] = std::numeric_limits<dfloat>::max();

  kernelInfo["includes"] += oklpath + "/findpts.okl.hpp";
  kernelInfo["includes"] += oklpath + "/poly.okl.hpp";

  auto findptsLocalKernel = platform->device.buildKernel(oklpath + "/findptsLocal.okl", "findptsLocal", kernelInfo);
  auto findptsLocalEvalKernel = platform->device.buildKernel(oklpath + "/findptsLocalEval.okl", "findptsLocalEval", kernelInfo);

  return {findptsLocalEvalKernel, findptsLocalKernel};
}
} // namespace findpts
