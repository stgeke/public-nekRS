#include "nrssys.hpp"
#include "kernelRequestManager.hpp"
#include "platform.hpp"
#include "fileUtils.hpp"

kernelRequestManager_t::kernelRequestManager_t(const platform_t& m_platform)
: kernelsProcessed(false),
  platformRef(m_platform)
{}

void
kernelRequestManager_t::add(const std::string& m_requestName,
                const std::string& m_fileName,
                const occa::properties& m_props,
                std::string m_suffix,
                bool checkUnique)
{
  this->add(kernelRequest_t{m_requestName, m_fileName, m_props, m_suffix}, checkUnique);
}
void
kernelRequestManager_t::add(kernelRequest_t request, bool checkUnique)
{
  auto iterAndBoolPair = kernels.insert(request);
  if(checkUnique)
  {
    int unique = (iterAndBoolPair.second) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &unique, 1, MPI_INT, MPI_MIN, platformRef.comm.mpiComm);
    nrsCheck(!unique, platformRef.comm.mpiComm, EXIT_FAILURE, 
             "Error in kernelRequestManager_t::add\nRequest details: %s\n", request.to_string().c_str());
  }

  const std::string fileName = request.fileName;
  fileNameToRequestMap[fileName].insert(request);
}
occa::kernel
kernelRequestManager_t::get(const std::string& request, bool checkValid) const
{
  if(checkValid){
    bool issueError = 0;
    issueError = !processed();
    issueError = (requestToKernelMap.count(request) == 0);

    int errorFlag = issueError ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &errorFlag, 1, MPI_INT, MPI_MAX, platformRef.comm.mpiComm);

    auto errTxt = [&]()
    { 
        std::stringstream txt;
        txt << "\n";
        txt << "Error in kernelRequestManager_t::getKernel():\n";
        txt << "Cannot find requested kernel " << request << "!\n";
        txt << "Available:\n";
        for(auto&& keyAndValue : requestToKernelMap)
          txt << "\t" << keyAndValue.first << "\n";

        txt << "===========================================================\n";
        return txt.str();
    };

    nrsCheck(errorFlag, platformRef.comm.mpiComm, EXIT_FAILURE, errTxt().c_str(), "");
  }

  return requestToKernelMap.at(request);
}

void
kernelRequestManager_t::compile()
{
  if(kernelsProcessed) return;

  kernelsProcessed = true;

  constexpr int maxCompilingRanks {20};

  const int rank = platform->cacheLocal ? platformRef.comm.localRank : platformRef.comm.mpiRank;
  const int ranksCompiling =
    std::min(
      maxCompilingRanks,
      platform->cacheLocal ?
        platformRef.comm.mpiCommLocalSize :
        platformRef.comm.mpiCommSize
    );


  std::vector<std::string> kernelFiles(fileNameToRequestMap.size());
  
  unsigned ctr = 0;
  for(auto&& fileNameAndRequests : fileNameToRequestMap)
  {
    kernelFiles[ctr] = fileNameAndRequests.first;
    ctr++;
  }

  const auto& device = platformRef.device;
  auto& requestToKernel = requestToKernelMap;
  auto& fileNameToRequest = fileNameToRequestMap;
  auto compileKernels = [&kernelFiles, &requestToKernel, &fileNameToRequest, &device, rank, ranksCompiling](){
    if(rank >= ranksCompiling) return;
    const unsigned nFiles = kernelFiles.size();
    for(unsigned fileId = 0; fileId < nFiles; ++fileId)
    {
      if(fileId % ranksCompiling == rank){
        const std::string fileName = kernelFiles[fileId];
        for(auto && kernelRequest : fileNameToRequest[fileName]){
          const std::string requestName = kernelRequest.requestName;
          const std::string fileName = kernelRequest.fileName;
          const std::string suffix = kernelRequest.suffix;
          const occa::properties props = kernelRequest.props;

          // MPI staging already handled
          auto kernel = device.buildKernel(fileName, props, suffix, false);
          requestToKernel[requestName] = kernel;
        }
      }
    }
  };

  const auto& kernelRequests = this->kernels;
  auto loadKernels = [&requestToKernel, &kernelRequests,&device](){
    for(auto&& kernelRequest : kernelRequests)
    {
      const std::string requestName = kernelRequest.requestName;
      if(requestToKernel.count(requestName) == 0){
        const std::string fileName = kernelRequest.fileName;
        const std::string suffix = kernelRequest.suffix;
        const occa::properties props = kernelRequest.props;

        // MPI staging already handled
        auto kernel = device.buildKernel(fileName, props, suffix, false);
        requestToKernel[requestName] = kernel;
      }
    }
  };

  MPI_Barrier(platform->comm.mpiComm);
  compileKernels();

  const auto OCCA_CACHE_DIR0 = occa::env::OCCA_CACHE_DIR;
  if(platform->cacheBcast) {
    const auto OCCA_CACHE_DIR_LOCAL = platform->tmpDir / fs::path("occa/");
    const auto srcPath = fs::path(std::string(getenv("OCCA_CACHE_DIR")));
    fileBcast(srcPath, OCCA_CACHE_DIR_LOCAL, platform->comm.mpiComm, platform->verbose); 
    occa::env::OCCA_CACHE_DIR = std::string(OCCA_CACHE_DIR_LOCAL);
  }

  MPI_Barrier(platform->comm.mpiComm);
  loadKernels();

  occa::env::OCCA_CACHE_DIR = OCCA_CACHE_DIR0;
}
