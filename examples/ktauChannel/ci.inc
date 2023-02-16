#include <math.h>

#include "constantFlowRate.hpp"

static int ciMode = 0;

static occa::kernel dragKernel;
static occa::kernel dragScaleKernel;

#define EPS 1e-2

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("POLYNOMIAL DEGREE", std::string("7"));
  options.setArgs("CUBATURE POLYNOMIAL DEGREE", std::string("10"));
  options.setArgs("CONSTANT FLOW RATE", "TRUE");
  options.setArgs("CONSTANT FLOW DIRECTION", "X");
  options.setArgs("RESTART FROM FILE", std::string("0"));
  options.setArgs("SOLUTION OUTPUT INTERVAL", "-1");
  options.setArgs("VISCOSITY", std::to_string(1.0 / 43500.0));
  options.setArgs("DENSITY", std::string("1.0"));
  options.setArgs("NUMBER TIMESTEPS", std::string("-1"));
  options.setArgs("END TIME", std::string("50"));
  options.setArgs("TIME INTEGRATOR", "TOMBO2");
  options.setArgs("ADVECTION TYPE", "CONVECTIVE+CUBATURE");
  options.setArgs("DT", std::string("5e-2"));
  options.setArgs("SUBCYCLING STEPS", std::string("0"));

  options.setArgs("PRESSURE KRYLOV SOLVER", "PGMRES+FLEXIBLE");
  options.setArgs("PRESSURE SOLVER TOLERANCE", std::string("1e-4"));
  options.setArgs("PRESSURE PRECONDITIONER","MULTIGRID");
  options.setArgs("PRESSURE MULTIGRID COARSE SOLVE", "TRUE");
  options.setArgs("PRESSURE MULTIGRID SMOOTHER", "FOURTHOPTCHEBYSHEV+ASM");
  options.setArgs("PRESSURE MULTIGRID CHEBYSHEV DEGREE", std::string("3"));
  options.setArgs("PRESSURE MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR",  std::string("1.1"));
  options.setArgs("PRESSURE INITIAL GUESS", "PROJECTION-ACONJ");
  
  options.setArgs("VELOCITY SOLVER TOLERANCE", std::string("1e-6"));
  options.setArgs("VELOCITY BLOCK SOLVER", "TRUE");
  options.setArgs("VELOCITY INITIAL GUESS","EXTRAPOLATION");

  options.setArgs("SCALAR00 SOLVER TOLERANCE", std::string("1e-6"));
  options.setArgs("SCALAR00 INITIAL GUESS","EXTRAPOLATION");

  options.setArgs("SCALAR01 SOLVER TOLERANCE", std::string("1e-8"));
  options.setArgs("SCALAR01 INITIAL GUESS","EXTRAPOLATION");

  options.setArgs("SCALAR02 SOLVER TOLERANCE", std::string("1e-6"));
  options.setArgs("SCALAR02 INITIAL GUESS","EXTRAPOLATION");
}

void buildCIKernel(occa::properties& kernelInfo)
{
  dragKernel = oudfBuildKernel(kernelInfo, "drag");
}

void ciTestErrors(nrs_t *nrs, dfloat time, int tstep, dfloat mue)
{
  if (!nrs->lastStep) return;

  mesh_t *mesh = nrs->meshV;

  //Friction factor
  const dfloat factor = 2.0 * ConstantFlowRate::scaleFactor();
  const dfloat fricFact = 2.0 * 2.27e-03;
  const double fricErr = (factor - fricFact)/fricFact;

  //Friction velocity
  const dfloat utauRef = 0.048; 
  occa::memory o_Sij = platform->o_mempool.slice2;
  nrs->SijOijKernel(mesh->Nelements,
		    nrs->fieldOffset,
		    0,
		    mesh->o_vgeo,
		    mesh->o_D,
		    nrs->o_U,
		    o_Sij);
  oogs::startFinish(o_Sij, 2*nrs->NVfields, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);
  
  occa::memory o_drag = platform->device.malloc(3 * nrs->fieldOffset * sizeof(dfloat));
  platform->linAlg->fillKernel(nrs->fieldOffset, 0.0, o_drag);

  //User specified boundary ID
  const int biD = 1;
  
  dragKernel(mesh->Nelements,
	     nrs->fieldOffset,
	     mue,
	     biD,
	     mesh->o_sgeo,
	     mesh->o_vmapM,
	     mesh->o_EToB,
	     mesh->o_invLMM,
	     o_Sij,
	     o_drag);

  const dfloat dragmax = platform->linAlg->max(mesh->Nlocal,o_drag,platform->comm.mpiComm);
  const dfloat utau = sqrt(dragmax);
  const dfloat utauErr = (utau-utauRef)/utauRef;
  
  if(platform->comm.mpiRank == 0){
    printf("Fiction Velocity: %g; Rel. error = %g\n",utau,utauErr);
    printf("Friction Factor: %g; Rel. error =%g\n",factor,fricErr);
  }

  (fabs(utauErr) < EPS && fabs(fricErr) < EPS) ? (CIPASS) : (CIFAIL);
}
