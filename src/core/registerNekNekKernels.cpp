#include <nrs.hpp>
#include <compileKernels.hpp>
#include <limits>

namespace {
// compute nearest power of two larger than v
// from: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
unsigned nearestPowerOfTwo(unsigned int v)
{
  unsigned answer = 1;
  while (answer < v)
    answer *= 2;
  return answer;
}
} // namespace

void registerNekNekKernels()
{
  dlong N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  const dlong Nq = N + 1;

  const std::string oklpath = getenv("NEKRS_KERNEL_DIR");

  std::string kernelName = "copyNekNekPoints";
  std::string fileName = oklpath + "/neknek/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, platform->kernelInfo);

  auto findptsKernelInfo = platform->kernelInfo;
  findptsKernelInfo["includes"].asArray();
  findptsKernelInfo["defines/p_D"] = 3;
  findptsKernelInfo["defines/p_Nq"] = Nq;
  findptsKernelInfo["defines/p_Np"] = Nq * Nq * Nq;
  findptsKernelInfo["defines/p_nptsBlock"] = 4;

  unsigned int Nq2 = Nq * Nq;
  const auto blockSize = nearestPowerOfTwo(Nq2);

  findptsKernelInfo["defines/p_blockSize"] = blockSize;
  findptsKernelInfo["defines/p_Nfp"] = Nq * Nq;
  findptsKernelInfo["defines/dlong"] = dlongString;
  findptsKernelInfo["defines/hlong"] = hlongString;
  findptsKernelInfo["defines/dfloat"] = dfloatString;
  findptsKernelInfo["defines/DBL_MAX"] = std::numeric_limits<dfloat>::max();

  findptsKernelInfo["includes"] += oklpath + "/findpts/findpts.okl.hpp";
  findptsKernelInfo["includes"] += oklpath + "/findpts/poly.okl.hpp";

  kernelName = "findptsLocal";
  fileName = oklpath + "/findpts/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, findptsKernelInfo);

  kernelName = "findptsLocalEval";
  fileName = oklpath + "/findpts/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, findptsKernelInfo);
}