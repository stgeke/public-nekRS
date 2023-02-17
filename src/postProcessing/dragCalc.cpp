#include <math.h>
#include <vector>
#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "nekInterfaceAdapter.hpp"
#include "postProcessing.hpp"

// private members
namespace {
  static dfloat *dragIntegral;

  static int nbID;
  static int NfpTotal;
  static dfloat mueLam;
  
  static occa::memory o_bID;
  static occa::memory o_drag;
  static occa::memory o_area;

  static occa::memory o_areab;
  static occa::memory o_dragb;
  
  static occa::kernel dragKernel;

  static bool setup = 0;
}

void postProcessing::buildDragKernel(occa::properties kernelInfo)
{
  const std::string path = getenv("NEKRS_KERNEL_DIR") + std::string("/postProcessing/");

  std::string fileName, kernelName;
  const std::string extension = ".okl";
  {
    kernelName = "drag";
    fileName = path + kernelName + extension;
    dragKernel = platform->device.buildKernel(fileName, kernelInfo, true);
  }
}

dfloat* postProcessing::getDrag()
{
  return dragIntegral;
}

void postProcessing::dragCalc(nrs_t *nrs, std::vector<int> bID, bool verbose)
{
  mesh_t *mesh = nrs->meshV;

  if(!setup){
    int bIDmin = *min_element(std::begin(bID), std::end(bID));
    nrsCheck(bIDmin < 1, platform->comm.mpiComm, EXIT_FAILURE,
	     "Boundary ID for drag calc must be >=1","");
    
    nbID = bID.size();
    o_bID = platform->device.malloc(nbID * sizeof(int), bID.data());

    NfpTotal = mesh->Nelements * mesh->Nfaces * mesh->Nfp;
    
    o_drag = platform->device.malloc(3 * NfpTotal * nbID * sizeof(dfloat));
    o_area = platform->device.malloc(NfpTotal * nbID * sizeof(dfloat));

    dragIntegral = (dfloat *)calloc(3 * nbID, sizeof(dfloat));

    platform->options.getArgs("VISCOSITY",mueLam);
  }
 
  platform->linAlg->fillKernel(3 * NfpTotal * nbID, 0.0, o_drag);
  platform->linAlg->fillKernel(NfpTotal * nbID, 0.0, o_area);

  occa::memory o_SijOij = platform->o_mempool.slice2;
  nrs->SijOijKernel(mesh->Nelements,
		    nrs->fieldOffset,
		    1,
		    mesh->o_vgeo,
		    mesh->o_D,
		    nrs->o_U,
		    o_SijOij);
  oogs::startFinish(o_SijOij, 9, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);

  dragKernel(mesh->Nelements,
	     nrs->fieldOffset,
	     NfpTotal,
	     nbID,
	     mueLam,
	     o_bID,
	     mesh->o_sgeo,
	     mesh->o_vmapM,
	     mesh->o_EToB,
	     mesh->o_invLMM,
	     o_SijOij,
	     o_drag,
	     o_area);
  
  for(int i=0; i < nbID; i++){
    const int boffset = i * NfpTotal;
    o_areab = o_area.slice(boffset * sizeof(dfloat), NfpTotal * sizeof(dfloat));
    const dfloat areasum = platform->linAlg->sum(NfpTotal, o_areab, platform->comm.mpiComm);

    if(!setup)nrsCheck(areasum==0, platform->comm.mpiComm, EXIT_FAILURE,
		       "Boundary ID %d not found in mesh",bID[i]);
    
    const int doffset = boffset * 3;
    for(int idim=0; idim < 3; idim++){
      o_dragb = o_drag.slice((doffset + idim*NfpTotal) * sizeof(dfloat), NfpTotal * sizeof(dfloat));
      dragIntegral[i * 3 + idim] = platform->linAlg->sum(NfpTotal, o_dragb, platform->comm.mpiComm) / areasum;
    }
  }

  if(platform->comm.mpiRank == 0 && verbose){
    for(int i=0; i < nbID; i++){
	printf("bid %d: dragx = %g, dragy = %g, dragz = %g\n",bID[i],dragIntegral[i*3],dragIntegral[i*3+1],dragIntegral[i*3+2]);
      }
  }
  setup = 1;
}
