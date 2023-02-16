#include <math.h>

#include "constantFlowRate.hpp"

static int ciMode = 0;

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

void ciTestErrors(nrs_t *nrs, dfloat time, int tstep)
{
  if (!nrs->lastStep) return;

  mesh_t *mesh = nrs->meshV;
  
  const dfloat factor = 2.0 * ConstantFlowRate::scaleFactor();

  const dfloat dnsFricFact = 2.0 * 2.27618e-03;

  const double facErr = (factor - dnsFricFact)/dnsFricFact;
  
  if(platform->comm.mpiRank == 0){
    printf("Rel. error in computed friction factor: %g\n",facErr);
  }

  (fabs(facErr) < EPS) ? (CIPASS) : (CIFAIL);
}
