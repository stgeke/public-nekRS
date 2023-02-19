#include <math.h>
#include <vector>
#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "nekInterfaceAdapter.hpp"
#include "postProcessing.hpp"

// private members
namespace {
  static dfloat dragIntegral;

  static int nbID;
  static int NfpTotal;
  static int Nblock;
  static dfloat mueLam;

  static dfloat *drag;
  static dfloat *area;
  
  static occa::memory o_bID;
  static occa::memory o_drag;
  static occa::memory o_area;

  static bool setup = 0;
}

dfloat postProcessing::getDrag()
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

    Nblock = (mesh->Nelements + BLOCKSIZE - 1) / BLOCKSIZE;

    drag = (dfloat *)calloc(Nblock * nbID, sizeof(dfloat));
    area = (dfloat *)calloc(Nblock * nbID, sizeof(dfloat));
    o_drag = platform->device.malloc(Nblock * nbID * sizeof(dfloat), drag);
    o_area = platform->device.malloc(Nblock * nbID * sizeof(dfloat), area);

    platform->options.getArgs("VISCOSITY",mueLam);
  }
 
  platform->linAlg->fillKernel(Nblock * nbID, 0.0, o_drag);
  platform->linAlg->fillKernel(Nblock * nbID, 0.0, o_area);

  occa::memory o_SijOij = platform->o_mempool.slice2;
  nrs->SijOijKernel(mesh->Nelements,
		    nrs->fieldOffset,
		    1,
		    mesh->o_vgeo,
		    mesh->o_D,
		    nrs->o_U,
		    o_SijOij);
  oogs::startFinish(o_SijOij, 9, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);

  auto dragKernel = platform->kernels.get("drag");
  
  dragKernel(mesh->Nelements,
	     nrs->fieldOffset,
	     NfpTotal,
	     nbID,
	     mueLam,
	     1,
	     mesh->o_sgeo,
	     mesh->o_vmapM,
	     mesh->o_EToB,
	     mesh->o_invLMM,
	     o_SijOij,
	     o_drag,
	     o_area);

  o_drag.copyTo(drag);
  o_area.copyTo(area);
  dfloat sbuf[2] = {0, 0};
  for (int n = 0; n < Nblock; n++){
    sbuf[0] += drag[n];
    sbuf[1] += area[n];
  }
  MPI_Allreduce(MPI_IN_PLACE, sbuf, 2, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
  
  dragIntegral = sbuf[0]/sbuf[1];
      
  setup = 1;
}
