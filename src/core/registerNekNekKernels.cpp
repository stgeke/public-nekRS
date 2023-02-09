#include <nrs.hpp>
#include <compileKernels.hpp>

void registerNekNekKernels()
{
  dlong N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  const dlong Nq = N + 1;

  findpts::findpts_t::initFindptsKernels(Nq);

  const std::string oklpath = getenv("NEKRS_KERNEL_DIR");

  const std::string kernelName = "copyNekNekPoints";
  const std::string fileName = oklpath + "/neknek/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, platform->kernelInfo);
}