#include <cfloat>
#include "bcMap.hpp"
#include "neknek.hpp"
#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "pointInterpolation.hpp"
#include <vector>

namespace {
void reserveAllocation(nrs_t *nrs, dlong npt)
{
  neknek_t *neknek = nrs->neknek;

  if (neknek->pointMap.size() && neknek->npt == npt)
    return;

  if (neknek->o_U.size()) {
    neknek->o_U.free();
  }

  if (neknek->o_S.size()) {
    neknek->o_S.free();
  }

  if (neknek->o_pointMap.size()) {
    neknek->o_pointMap.free();
  }

  // compute page-aligned fieldOffset
  neknek->fieldOffset = npt;
  const int pageW = ALIGN_SIZE / sizeof(dfloat);
  if (neknek->fieldOffset % pageW)
    neknek->fieldOffset = (neknek->fieldOffset / pageW + 1) * pageW;

  neknek->pointMap.resize(nrs->fieldOffset + 1);
  neknek->o_pointMap = platform->device.malloc((nrs->fieldOffset + 1) * sizeof(dlong));

  if (npt) {
    neknek->o_U =
        platform->device.malloc(nrs->NVfields * neknek->fieldOffset * (neknek->nEXT + 1) * sizeof(dfloat));
    if (neknek->Nscalar) {
      neknek->o_S = platform->device.malloc(neknek->Nscalar * neknek->fieldOffset * (neknek->nEXT + 1) *
                                            sizeof(dfloat));
    }
    else {
      neknek->o_S = platform->device.malloc(1 * sizeof(dfloat));
    }
  }
  else {
    neknek->o_U = platform->device.malloc(1 * sizeof(dfloat));
    neknek->o_S = platform->device.malloc(1 * sizeof(dfloat));
  }
  neknek->npt = npt;
}

void checkValidBoundaryConditions(nrs_t *nrs)
{
  if (!nrs->cds)
    return;

  auto *cds = nrs->cds;
  auto *mesh = nrs->meshV;
  auto *neknek = nrs->neknek;

  std::vector<dlong> missingInterpBound(neknek->Nscalar, 0);
  std::vector<dlong> extraInterpBound(neknek->Nscalar, 0);
  for (int s = 0; s < neknek->Nscalar; ++s) {
    if (!cds->compute[s])
      continue;

    auto *EToB = cds->EToB[s];
    for (dlong pt = 0; pt < mesh->Nelements * mesh->Nfaces; ++pt) {
      missingInterpBound[s] |= (nrs->EToB[pt] == bcMap::bcTypeINT && EToB[pt] != bcMap::bcTypeINTS);
      extraInterpBound[s] |= (!nrs->EToB[pt] == bcMap::bcTypeINT && EToB[pt] == bcMap::bcTypeINTS);
    }
  }

  MPI_Allreduce(MPI_IN_PLACE,
                missingInterpBound.data(),
                neknek->Nscalar,
                MPI_DLONG,
                MPI_MAX,
                platform->comm.mpiComm);
  MPI_Allreduce(MPI_IN_PLACE,
                extraInterpBound.data(),
                neknek->Nscalar,
                MPI_DLONG,
                MPI_MAX,
                platform->comm.mpiComm);
  bool issueError = false;
  for (int s = 0; s < neknek->Nscalar; ++s) {
    bool invalid = missingInterpBound[s];
    if (platform->comm.mpiRank == 0 && invalid) {
      std::cout << "Error: scalar " << s
                << " has a non-interpolating boundary condition where the fluid does!\n";
    }
    issueError |= invalid;

    invalid = extraInterpBound[s];
    if (platform->comm.mpiRank == 0 && invalid) {
      std::cout << "Error: scalar " << s
                << " has an interpolating boundary condition where the fluid does not!\n";
    }
    issueError |= invalid;
  }

  if (issueError) {
    ABORT(EXIT_FAILURE);
  }
}

void updateInterpPoints(nrs_t *nrs)
{
  // called in case of moving mesh ONLY
  if (!nrs->neknek->globalMovingMesh)
    return;

  auto *neknek = nrs->neknek;
  const dlong nsessions = neknek->nsessions;
  const dlong sessionID = neknek->sessionID;

  auto *mesh = nrs->meshV;

  // Setup findpts
  const dfloat tol = 5e-13;
  constexpr dlong npt_max = 128;
  const dfloat bb_tol = 0.01;

  auto &device = platform->device.occaDevice();

  // TODO: possible to cache this in moving mesh case?
  std::vector<std::shared_ptr<pointInterpolation_t>> sessionInterpolators(nsessions);
  for (dlong i = 0; i < nsessions; ++i) {
    sessionInterpolators[i] = std::make_shared<pointInterpolation_t>(nrs, tol, i == sessionID);
    sessionInterpolators[i]->setTimerLevel(TimerLevel::Basic);
    sessionInterpolators[i]->setTimerName("neknek_t::");
  }

  neknek->interpolator.reset();
  neknek->interpolator = std::make_shared<pointInterpolation_t>(nrs, tol);
  neknek->interpolator->setTimerLevel(TimerLevel::Basic);
  neknek->interpolator->setTimerName("neknek_t::");

  // neknekX[:] = mesh->x[pointMap[:]]
  neknek->copyNekNekPointsKernel(mesh->Nlocal,
                                 neknek->o_pointMap,
                                 mesh->o_x,
                                 mesh->o_y,
                                 mesh->o_z,
                                 neknek->o_x,
                                 neknek->o_y,
                                 neknek->o_z);

  // add points (use GPU version)
  for (dlong sess = 0; sess < nsessions; ++sess) {
    const auto nPoint = (sess == sessionID) ? 0 : neknek->npt;
    sessionInterpolators[sess]->addPoints(nPoint, neknek->o_x, neknek->o_y, neknek->o_z);
  }

  neknek->interpolator->addPoints(neknek->npt, neknek->o_x, neknek->o_y, neknek->o_z);

  constexpr bool printWarnings = true;
  for (dlong sess = 0; sess < nsessions; ++sess) {
    sessionInterpolators[sess]->find(printWarnings);
  }

  auto &sessionData = neknek->interpolator->data();

  // TODO: possible to move to GPU?
  // copy results from other session into the point interpolator
  for (dlong sess = 0; sess < nsessions; ++sess) {
    auto data = sessionInterpolators[sess]->data();
    const auto nPoint = (sess == sessionID) ? 0 : neknek->npt;
    for (dlong pt = 0; pt < nPoint; ++pt) {
      sessionData.code[pt] = data.code[pt];
      sessionData.proc[pt] = data.proc[pt];
      sessionData.el[pt] = data.el[pt];

      sessionData.r[3 * pt + 0] = data.r[3 * pt + 0];
      sessionData.r[3 * pt + 1] = data.r[3 * pt + 1];
      sessionData.r[3 * pt + 2] = data.r[3 * pt + 2];

      sessionData.dist2[pt] = data.dist2[pt];
    }
  }
}

dlong computeNumInterpPoints(nrs_t *nrs)
{
  auto *mesh = nrs->meshV;
  dlong numInterpFaces = 0;
  for (dlong e = 0; e < mesh->Nelements; ++e) {
    for (dlong f = 0; f < mesh->Nfaces; ++f) {
      numInterpFaces += (nrs->EToB[f + mesh->Nfaces * e] == bcMap::bcTypeINT);
    }
  }
  return numInterpFaces * mesh->Nfp;
}

void findInterpPoints(nrs_t *nrs)
{

  auto *neknek = nrs->neknek;
  const dlong nsessions = neknek->nsessions;
  const dlong sessionID = neknek->sessionID;

  auto *mesh = nrs->meshV;

  // Setup findpts
  const dfloat tol = 5e-13;
  constexpr dlong npt_max = 128;
  const dfloat bb_tol = 0.01;

  auto &device = platform->device.occaDevice();

  // TODO: possible to cache this in moving mesh case?
  std::vector<std::shared_ptr<pointInterpolation_t>> sessionInterpolators(nsessions);
  for (dlong i = 0; i < nsessions; ++i) {
    sessionInterpolators[i] = std::make_shared<pointInterpolation_t>(nrs, tol, i == sessionID);
  }

  neknek->interpolator.reset();
  neknek->interpolator = std::make_shared<pointInterpolation_t>(nrs, tol);

  auto numPoints = computeNumInterpPoints(nrs);
  reserveAllocation(nrs, numPoints);

  std::vector<dfloat> neknekX(numPoints, 0.0);
  std::vector<dfloat> neknekY(numPoints, 0.0);
  std::vector<dfloat> neknekZ(numPoints, 0.0);

  dlong ip = 0;
  std::fill(neknek->pointMap.begin(), neknek->pointMap.end(), -1);
  for (dlong e = 0; e < mesh->Nelements; ++e) {
    for (dlong f = 0; f < mesh->Nfaces; ++f) {

      for (dlong m = 0; m < mesh->Nfp; ++m) {
        dlong id = mesh->Nfaces * mesh->Nfp * e + mesh->Nfp * f + m;
        dlong idM = mesh->vmapM[id];

        if (nrs->EToB[f + mesh->Nfaces * e] == bcMap::bcTypeINT) {
          neknekX[ip] = mesh->x[idM];
          neknekY[ip] = mesh->y[idM];
          neknekZ[ip] = mesh->z[idM];

          neknek->pointMap[idM] = ip;
          ++ip;
        }
      }
    }
  }
  neknek->pointMap[nrs->fieldOffset] = neknek->fieldOffset;
  neknek->o_pointMap.copyFrom(neknek->pointMap.data());

  // check: all computed scalars must have `int` b.c. if the fluid has `int` b.c.
  checkValidBoundaryConditions(nrs);

  // add points
  for (dlong sess = 0; sess < nsessions; ++sess) {
    const auto nPoint = (sess == sessionID) ? 0 : numPoints;
    sessionInterpolators[sess]->addPoints(nPoint, neknekX.data(), neknekY.data(), neknekZ.data());
  }

  neknek->interpolator->addPoints(numPoints, neknekX.data(), neknekY.data(), neknekZ.data());

  constexpr bool printWarnings = true;
  for (dlong sess = 0; sess < nsessions; ++sess) {
    sessionInterpolators[sess]->find(printWarnings);
  }

  auto &sessionData = neknek->interpolator->data();

  // copy results from other session into the point interpolator
  for (dlong sess = 0; sess < nsessions; ++sess) {
    auto data = sessionInterpolators[sess]->data();
    const auto nPoint = (sess == sessionID) ? 0 : numPoints;
    for (dlong pt = 0; pt < nPoint; ++pt) {
      sessionData.code[pt] = data.code[pt];
      sessionData.proc[pt] = data.proc[pt];
      sessionData.el[pt] = data.el[pt];

      sessionData.r[3 * pt + 0] = data.r[3 * pt + 0];
      sessionData.r[3 * pt + 1] = data.r[3 * pt + 1];
      sessionData.r[3 * pt + 2] = data.r[3 * pt + 2];

      sessionData.dist2[pt] = data.dist2[pt];
    }
  }

  // allocate device coordinates for later use
  if (neknek->globalMovingMesh) {
    neknek->o_x = platform->device.malloc(neknek->npt * sizeof(dfloat), neknekX.data());
    neknek->o_y = platform->device.malloc(neknek->npt * sizeof(dfloat), neknekY.data());
    neknek->o_z = platform->device.malloc(neknek->npt * sizeof(dfloat), neknekZ.data());
  }
}

void neknekSetup(nrs_t *nrs)
{
  neknek_t *neknek = nrs->neknek;

  // determine if sessions are coupled
  auto numInterpPoints = computeNumInterpPoints(nrs);

  dlong numInterpPointsSession = numInterpPoints;
  MPI_Allreduce(MPI_IN_PLACE, &numInterpPointsSession, 1, MPI_DLONG, MPI_SUM, platform->comm.mpiComm);

  dlong minPointsAcrossSessions = numInterpPointsSession;
  MPI_Allreduce(MPI_IN_PLACE, &minPointsAcrossSessions, 1, MPI_DLONG, MPI_MIN, platform->comm.mpiCommParent);

  dlong maxPointsAcrossSessions = numInterpPointsSession;
  MPI_Allreduce(MPI_IN_PLACE, &maxPointsAcrossSessions, 1, MPI_DLONG, MPI_MAX, platform->comm.mpiCommParent);

  neknek->coupled = minPointsAcrossSessions > 0;

  if ((minPointsAcrossSessions == 0) && (maxPointsAcrossSessions > 0)) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "Error: one session has no interpolation points, but another session does!\n";
    }
    ABORT(EXIT_FAILURE);
  }

  if (!neknek->coupled) {
    neknek->Nscalar = nrs->Nscalar;
    reserveAllocation(nrs, 0);
    neknek->pointMap[nrs->fieldOffset] = 0;
    neknek->o_pointMap.copyFrom(neknek->pointMap.data());
    return;
  }

  if (platform->options.compareArgs("CONSTANT FLOW RATE", "TRUE")) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "Neknek + constant flow rate is not currently implemented!\n";
    }
    ABORT(EXIT_FAILURE);
  }

  const dlong nsessions = neknek->nsessions;

  MPI_Comm globalComm = neknek->globalComm;
  dlong globalRank;
  MPI_Comm_rank(globalComm, &globalRank);

  if(globalRank == 0) printf("configuring neknek with %d sessions\n", nsessions);

  dlong nFields[2];
  nFields[0] = nrs->Nscalar;
  nFields[1] = -nFields[0];
  MPI_Allreduce(MPI_IN_PLACE, nFields, 2, MPI_DLONG, MPI_MIN, globalComm);
  nFields[1] = -nFields[1];
  if (nFields[0] != nFields[1]) {
    if(globalRank == 0) {
      std::cout << "WARNING: varying numbers of scalars; only updating " << nFields[0] << std::endl;
    }
  }
  neknek->Nscalar = nFields[0];

  const dlong movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  dlong globalMovingMesh;
  MPI_Allreduce(&movingMesh, &globalMovingMesh, 1, MPI_DLONG, MPI_MAX, globalComm);
  neknek->globalMovingMesh = globalMovingMesh;

  findInterpPoints(nrs);
}

} // namespace

neknek_t::neknek_t(nrs_t *nrs, const session_data_t &session)
    : nsessions(session.nsessions), sessionID(session.sessionID), globalComm(session.globalComm),
      localComm(session.localComm), coupled(session.coupled)
{

  nrs->neknek = this;
  if (nrs->cds) {
    nrs->cds->neknek = this;
  }

  platform->options.getArgs("BOUNDARY EXTRAPOLATION ORDER", this->nEXT);

  this->coeffEXT.resize(this->nEXT);
  this->o_coeffEXT = platform->device.malloc(this->nEXT * sizeof(dfloat));

  neknekSetup(nrs);

  // variable p0th + nek-nek is not supported
  if (this->coupled) {
    int issueError = 0;
    if (nrs->pSolver->allNeumann && platform->options.compareArgs("LOWMACH", "TRUE")) {
      issueError = 1;
    }

    MPI_Allreduce(MPI_IN_PLACE, &issueError, 1, MPI_DLONG, MPI_MAX, platform->comm.mpiCommParent);
    if (issueError) {
      if (platform->comm.mpiRank == 0) {
        printf("ERROR: neknek + variable p0th is not supported!\n");
      }
      ABORT(EXIT_FAILURE);
    }
  }

  this->copyNekNekPointsKernel = platform->kernels.get("copyNekNekPoints");
}

void neknek_t::updateBoundary(nrs_t *nrs, int tstep, int stage)
{
  if (!this->coupled)
    return;

  // do not invoke barrier -- this is performed later
  platform->timer.tic("neknek update boundary", 0);

  // do not invoke barrier in timer_t::tic
  platform->timer.tic("neknek sync", 0);
  MPI_Barrier(platform->comm.mpiCommParent);
  platform->timer.toc("neknek sync");
  this->tSync = platform->timer.query("neknek sync", "HOST:MAX");

  if (this->globalMovingMesh) {
    platform->timer.tic("neknek updateInterpPoints", 1);
    updateInterpPoints(nrs);
    platform->timer.toc("neknek updateInterpPoints");
  }

  platform->timer.tic("neknek exchange", 1);

  this->interpolator->eval(nrs->NVfields, nrs->fieldOffset, nrs->o_U, this->fieldOffset, this->o_U);

  if (this->Nscalar) {
    this->interpolator->eval(this->Nscalar, nrs->fieldOffset, nrs->cds->o_S, this->fieldOffset, this->o_S);
  }

  // lag state, update timestepper coefficients and compute extrapolated state
  if (stage == 1) {
    auto *mesh = nrs->meshV;
    int extOrder = std::min(tstep, this->nEXT);
    int bdfOrder = std::min(tstep, nrs->nBDF);
    nek::extCoeff(this->coeffEXT.data(), nrs->dt, extOrder, bdfOrder);

    for (int i = this->nEXT; i > extOrder; i--)
      this->coeffEXT[i - 1] = 0.0;

    this->o_coeffEXT.copyFrom(this->coeffEXT.data(), this->nEXT * sizeof(dfloat));

    for (int s = this->nEXT + 1; s > 1; s--) {
      auto Nbyte = nrs->NVfields * this->fieldOffset * sizeof(dfloat);
      this->o_U.copyFrom(this->o_U, Nbyte, (s - 1) * Nbyte, (s - 2) * Nbyte);

      Nbyte = this->Nscalar * this->fieldOffset * sizeof(dfloat);
      this->o_S.copyFrom(this->o_S, Nbyte, (s - 1) * Nbyte, (s - 2) * Nbyte);
    }

    auto o_Uold = this->o_U + this->fieldOffset * nrs->NVfields * sizeof(dfloat);
    auto o_Sold = this->o_S + this->fieldOffset * this->Nscalar * sizeof(dfloat);

    nrs->extrapolateKernel(this->npt,
                           nrs->NVfields,
                           this->nEXT,
                           this->fieldOffset,
                           this->o_coeffEXT,
                           o_Uold,
                           this->o_U);

    if (this->Nscalar) {
      nrs->extrapolateKernel(this->npt,
                             this->Nscalar,
                             this->nEXT,
                             this->fieldOffset,
                             this->o_coeffEXT,
                             o_Sold,
                             this->o_S);
    }
  }

  platform->timer.toc("neknek exchange");

  this->tExch = platform->timer.query("neknek exchange", "DEVICE:MAX");
  this->ratio = this->tSync / this->tExch;

  platform->timer.toc("neknek update boundary");
}
