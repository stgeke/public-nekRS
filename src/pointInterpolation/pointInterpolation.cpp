
#include <cstdlib>
#include <mpi.h>
#include "nrs.hpp"
#include "platform.hpp"
#include <vector>

#include "findpts.hpp"

#include "pointInterpolation.hpp"
#include <algorithm>

pointInterpolation_t::pointInterpolation_t(nrs_t *nrs_, double newton_tol_, bool mySession_)
    : pointInterpolation_t(nrs_, nrs_->meshV->Nlocal, nrs_->meshV->Nlocal, newton_tol_, mySession_)
{
}

pointInterpolation_t::pointInterpolation_t(nrs_t *nrs_,
                                           dlong localHashSize,
                                           dlong globalHashSize,
                                           double newton_tol_,
                                           bool mySession_)
    : nrs(nrs_), newton_tol(newton_tol_), mySession(mySession_), nPoints(0)
{

  newton_tol = std::max(5e-13, newton_tol_);

  const int npt_max = 128;
  const dfloat bb_tol = 0.01;

  mesh_t *mesh = nrs->meshV;

  if (mySession) {
    mesh->o_x.copyTo(mesh->x, mesh->Nlocal * sizeof(dfloat));
    mesh->o_y.copyTo(mesh->y, mesh->Nlocal * sizeof(dfloat));
    mesh->o_z.copyTo(mesh->z, mesh->Nlocal * sizeof(dfloat));
  }

  findpts_ = std::make_unique<findpts::findpts_t>(platform->comm.mpiCommParent,
                                                  mySession ? mesh->x : nullptr,
                                                  mySession ? mesh->y : nullptr,
                                                  mySession ? mesh->z : nullptr,
                                                  mesh->Nq,
                                                  mySession ? mesh->Nelements : 0,
                                                  2 * mesh->Nq,
                                                  bb_tol,
                                                  localHashSize,
                                                  globalHashSize,
                                                  npt_max,
                                                  newton_tol);
}

void pointInterpolation_t::find(bool printWarnings)
{
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic("pointInterpolation_t::find", 1);
  }

  int iErr = 0;
  iErr += !pointsAdded;
  MPI_Allreduce(MPI_IN_PLACE, &iErr, 1, MPI_DLONG, MPI_MAX, platform->comm.mpiComm);
  if (iErr) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "pointInterpolation_t::find called without any points added!\n";
    }
  }
  
  nrsCheck(iErr, platform->comm.mpiComm, EXIT_FAILURE, "", "");

  const auto n = nPoints;

  if (useHostPoints) {
    findpts_->find(&data_, _x, _y, _z, n);
  }
  else {
    findpts_->find(&data_, _o_x, _o_y, _o_z, n);
  }

  if (printWarnings) {

    auto *h_x = _x;
    auto *h_y = _y;
    auto *h_z = _z;
    if (useDevicePoints) {
      h_x = h_x_vec.data();
      h_y = h_y_vec.data();
      h_z = h_z_vec.data();
      _o_x.copyTo(h_x, n * sizeof(dfloat));
      _o_y.copyTo(h_y, n * sizeof(dfloat));
      _o_z.copyTo(h_z, n * sizeof(dfloat));
    }

    dlong nFail = 0;
    for (int in = 0; in < n; ++in) {
      if (data_.code_base[in] == findpts::CODE_BORDER) {
        if (data_.dist2_base[in] > 10 * newton_tol) {
          nFail += 1;
          if (nFail < 5) {
            std::cout << " WARNING: point on boundary or outside the mesh xy[z]d^2: " << h_x[in] << ","
                      << h_y[in] << ", " << h_z[in] << ", " << data_.dist2_base[in] << std::endl;
          }
        }
      }
      else if (data_.code_base[in] == findpts::CODE_NOT_FOUND) {
        nFail += 1;
        if (nFail < 5) {
          std::cout << " WARNING: point not within mesh xy[z]: " << h_x[in] << "," << h_y[in] << ", "
                    << h_z[in] << std::endl;
        }
      }
    }
    hlong counts[4] = {n, nFail, 0, 0};
    MPI_Reduce(counts, counts + 2, 2, MPI_HLONG, MPI_SUM, 0, platform_t::getInstance()->comm.mpiComm);
    if (platform_t::getInstance()->comm.mpiRank == 0 && counts[3] > 0) {
      std::cout << "interp::find - Total number of points = " << counts[2] << ", failed = " << counts[3]
                << std::endl;
    }
  }

  if (timerLevel != TimerLevel::None) {
    platform->timer.toc("pointInterpolation_t::find");
  }
}

void pointInterpolation_t::eval(dlong nFields,
                                dlong inputFieldOffset,
                                occa::memory o_in,
                                dlong outputFieldOffset,
                                occa::memory o_out)
{
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic("pointInterpolation_t::eval", 1);
  }

  const auto n = data_.code.size();
  findpts_->eval(n, nFields, inputFieldOffset, outputFieldOffset, o_in, &data_, o_out);

  if (timerLevel != TimerLevel::None) {
    platform->timer.toc("pointInterpolation_t::eval");
  }
}

void pointInterpolation_t::eval(dlong nFields,
                                dlong inputFieldOffset,
                                dfloat *in,
                                dlong outputFieldOffset,
                                dfloat *out)
{
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic("pointInterpolation_t::eval", 1);
  }

  const auto n = data_.code.size();
  findpts_->eval(n, nFields, inputFieldOffset, outputFieldOffset, in, &data_, out);

  if (timerLevel != TimerLevel::None) {
    platform->timer.toc("pointInterpolation_t::eval");
  }
}

void pointInterpolation_t::addPoints(int n, dfloat * x, dfloat * y, dfloat * z)
{

  pointsAdded = true;
  useHostPoints = true;
  useDevicePoints = false;

  if(n > nPoints){
    data_ = findpts::data_t(n);
  }

  nPoints = n;

  _x = x;
  _y = y;
  _z = z;
}

void pointInterpolation_t::addPoints(int n, occa::memory o_x, occa::memory o_y, occa::memory o_z)
{

  pointsAdded = true;
  useHostPoints = false;
  useDevicePoints = true;

  if (n > nPoints) {
    data_ = findpts::data_t(n);
  }

  nPoints = n;

  _o_x = o_x;
  _o_y = o_y;
  _o_z = o_z;

  h_x_vec.resize(n);
  h_y_vec.resize(n);
  h_z_vec.resize(n);
}

void pointInterpolation_t::setTimerLevel(TimerLevel level)
{
  timerLevel = level;
  findpts_->setTimerLevel(level);
}

TimerLevel pointInterpolation_t::getTimerLevel() const { return timerLevel; }

void pointInterpolation_t::setTimerName(std::string name)
{
  timerName = name;
  findpts_->setTimerName(name);
}
