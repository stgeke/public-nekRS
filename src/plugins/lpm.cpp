#include "nekInterfaceAdapter.hpp" // for nek::coeffAB
#include "lpm.hpp"
#include "nrs.hpp"
#include "pointInterpolation.hpp"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <regex>

namespace {
std::string lowerCase(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

int computeFieldOffset(int n)
{
  auto offset = n;
  const int pageW = ALIGN_SIZE / sizeof(dfloat);
  if (offset % pageW)
    offset = (offset / pageW + 1) * pageW;
  return offset;
}
} // namespace

lpm_t::lpm_t(nrs_t *nrs_, dfloat newton_tol_)
    : nrs(nrs_), nAB(nrs->nEXT), newton_tol(newton_tol_),
      interp(std::make_unique<pointInterpolation_t>(nrs, newton_tol))
{
  coeffAB.resize(nAB);
  o_coeffAB = platform->device.malloc(nAB * sizeof(dfloat));

  // coordinates are registered by default
  registerDOF("x");
  registerDOF("y");
  registerDOF("z");

  nStagesSumManyKernel = platform->kernels.get("nStagesSumMany");
  remapParticlesKernel = platform->kernels.get("remapParticles");

  setTimerLevel(timerLevel);
  setTimerName(timerName);
}

lpm_t::lpm_t(nrs_t *nrs_, int nAB_, dfloat newton_tol_)
    : nrs(nrs_), nAB(nAB_), newton_tol(newton_tol_),
      interp(std::make_unique<pointInterpolation_t>(nrs, newton_tol))
{
  coeffAB.resize(nAB);
  o_coeffAB = platform->device.malloc(nAB * sizeof(dfloat));

  // coordinates are registered by default
  registerDOF("x");
  registerDOF("y");
  registerDOF("z");

  nStagesSumManyKernel = platform->kernels.get("nStagesSumMany");
  remapParticlesKernel = platform->kernels.get("remapParticles");

  setTimerLevel(timerLevel);
  setTimerName(timerName);
}

void lpm_t::registerDOF(std::string dofName, bool output) { registerDOF(1, dofName, output); }

void lpm_t::registerDOF(dlong Nfields, std::string dofName, bool output)
{
  if (this->constructed()) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: cannot register DOF " << dofName << " after construction!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }

  dofName = lowerCase(dofName);
  const auto nDOFs = dofIds.size();
  if (dofIds.count(dofName) == 0) {
    dofIds[dofName] = nDOFs;
    outputDofs[dofName] = output;
    dofCounts[dofName] = Nfields;
    nDOFs_ += Nfields;
    fieldType[dofName] = FieldType::DOF;
  }
}

int lpm_t::dofId(std::string dofName) const
{
  dofName = lowerCase(dofName);
  if (dofIds.count(dofName) == 0) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: DOF " << dofName << " not registered!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }
  return dofIds.at(dofName);
}

int lpm_t::numDOFs(std::string dofName) const
{
  dofName = lowerCase(dofName);
  if (dofIds.count(dofName) == 0) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: DOF " << dofName << " not registered!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }
  return dofCounts.at(dofName);
}

void lpm_t::registerProp(std::string propName, bool output) { registerProp(1, propName, output); }

void lpm_t::registerProp(dlong Nfields, std::string propName, bool output)
{
  if (this->constructed()) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: cannot register prop " << propName << " after construction!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }

  propName = lowerCase(propName);
  const auto nprops = propIds.size();
  if (propIds.count(propName) == 0) {
    propIds[propName] = nprops;
    outputProps[propName] = output;
    propCounts[propName] = Nfields;
    nProps_ += Nfields;
    fieldType[propName] = FieldType::PROP;
  }
}

int lpm_t::propId(std::string propName) const
{
  propName = lowerCase(propName);
  if (propIds.count(propName) == 0) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: prop " << propName << " not registered!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }
  return propIds.at(propName);
}

int lpm_t::numProps(std::string propName) const
{
  propName = lowerCase(propName);
  if (propIds.count(propName) == 0) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: prop " << propName << " not registered!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }
  return propCounts.at(propName);
}

void lpm_t::registerInterpField(std::string interpFieldName, int Nfields, occa::memory o_fld, bool output)
{
  if (this->constructed()) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: cannot register interpField " << interpFieldName << " after construction!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }

  interpFieldName = lowerCase(interpFieldName);
  const auto nInterpFields = interpFieldIds.size();
  if (interpFieldIds.count(interpFieldName) == 0) {
    interpFieldIds[interpFieldName] = nInterpFields;
    interpFieldCounts[interpFieldName] = Nfields;
    outputInterpFields[interpFieldName] = output;
    interpFieldInputs[interpFieldName] = o_fld;
    nInterpFields_ += Nfields;
    fieldType[interpFieldName] = FieldType::INTERP_FIELD;
  }
}

void lpm_t::registerInterpField(std::string interpFieldName, occa::memory o_fld, bool output)
{
  registerInterpField(interpFieldName, 1, o_fld, output);
}

int lpm_t::interpFieldId(std::string interpFieldName) const
{
  interpFieldName = lowerCase(interpFieldName);
  if (interpFieldIds.count(interpFieldName) == 0) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: interpField " << interpFieldName << " not registered!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }
  return interpFieldIds.at(interpFieldName);
}

int lpm_t::numFieldsInterp(std::string interpFieldName) const
{
  interpFieldName = lowerCase(interpFieldName);
  if (interpFieldIds.count(interpFieldName) == 0) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: interpField " << interpFieldName << " not registered!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }
  return interpFieldCounts.at(interpFieldName);
}

void lpm_t::setUserRHS(lpm_t::rhsFunc_t userRHS) { userRHS_ = userRHS; }

void lpm_t::addUserData(void *userdata) { userdata_ = userdata; }

occa::memory lpm_t::getDOF(int dofId) { return o_y + dofId * fieldOffset_ * sizeof(dfloat); }
occa::memory lpm_t::getDOF(std::string dofName) { return getDOF(dofId(dofName)); }

std::vector<dfloat> lpm_t::getDOFHost(std::string dofName)
{
  auto o_dof = getDOF(dofName);
  auto Nfields = numDOFs(dofName);

  std::vector<dfloat> h_dof(Nfields * fieldOffset_);
  o_dof.copyTo(h_dof.data(), Nfields * fieldOffset_ * sizeof(dfloat));
  return h_dof;
}

occa::memory lpm_t::getProp(int propId) { return o_prop + propId * fieldOffset_ * sizeof(dfloat); }
occa::memory lpm_t::getProp(std::string propName) { return getProp(propId(propName)); }

std::vector<dfloat> lpm_t::getPropHost(std::string propName)
{
  auto o_propEntry = getProp(propName);
  auto Nfields = numProps(propName);

  std::vector<dfloat> h_prop(Nfields * fieldOffset_);
  o_propEntry.copyTo(h_prop.data(), Nfields * fieldOffset_ * sizeof(dfloat));
  return h_prop;
}

occa::memory lpm_t::getInterpField(int interpFieldId)
{
  return o_interpFld + interpFieldId * fieldOffset_ * sizeof(dfloat);
}
occa::memory lpm_t::getInterpField(std::string interpFieldName)
{
  return getInterpField(interpFieldId(interpFieldName));
}

std::vector<dfloat> lpm_t::getInterpFieldHost(std::string interpFieldName)
{
  auto o_interpFldEntry = getInterpField(interpFieldName);
  auto Nfields = numFieldsInterp(interpFieldName);

  std::vector<dfloat> h_interpField(Nfields * fieldOffset_);
  o_interpFldEntry.copyTo(h_interpField.data(), Nfields * fieldOffset_ * sizeof(dfloat));
  return h_interpField;
}

void lpm_t::construct(int nParticles)
{
  nParticles_ = nParticles;
  fieldOffset_ = computeFieldOffset(nParticles);

  o_y = platform->device.malloc(fieldOffset_ * nDOFs_ * sizeof(dfloat));
  o_ydot = platform->device.malloc(nAB * fieldOffset_ * nDOFs_ * sizeof(dfloat));

  if (nProps_) {
    o_prop = platform->device.malloc(fieldOffset_ * nProps_ * sizeof(dfloat));
  }
  if (nInterpFields_) {
    o_interpFld = platform->device.malloc(fieldOffset_ * nInterpFields_ * sizeof(dfloat));
  }

  constructed_ = true;
}

void lpm_t::coeff(dfloat *dt, int tstep)
{
  const int order = std::min(tstep, this->nAB);
  nek::coeffAB(coeffAB.data(), dt, order);
  for (int i = 0; i < order; ++i)
    coeffAB[i] *= dt[0];
  for (int i = order; i > order; i--)
    coeffAB[i - 1] = 0.0;
  o_coeffAB.copyFrom(coeffAB.data(), nAB * sizeof(dfloat));
}

void lpm_t::interpolate()
{
  for (auto [interpFieldName, interpFieldId] : interpFieldInputs) {
    interpolate(interpFieldName);
  }
}

void lpm_t::interpolate(std::string interpFieldName)
{
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic(timerName + "interpolate", 1);
  }
  auto o_fld = interpFieldInputs.at(interpFieldName);
  auto o_interpFld = getInterpField(interpFieldName);
  const auto Nfields = numFieldsInterp(interpFieldName);

  interp->eval(Nfields, nrs->fieldOffset, o_fld, fieldOffset_, o_interpFld);
  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "interpolate");
  }
}

void lpm_t::integrate(dfloat t0, dfloat tf, int step)
{
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic(timerName + "integrate", 1);
  }

  if (platform->options.compareArgs("MOVING MESH", "TRUE")) {
    interp.reset();
    interp = std::make_unique<pointInterpolation_t>(nrs, newton_tol);
  }

  if (!constructed_) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: cannot integrate before construction!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }

  if (!userRHS_) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: cannot integrate without setting userRHS!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }

  coeff(nrs->dt, step);

  auto o_xcoord = getDOF("x");
  auto o_ycoord = getDOF("y");
  auto o_zcoord = getDOF("z");

  interp->addPoints(size(), o_xcoord, o_ycoord, o_zcoord);

  platform->timer.tic("lpm_t::find", 1);
  interp->find(issueWarnings_);
  platform->timer.toc("lpm_t::find");

  deleteParticles();

  if (userODESolver_) {
    userODESolver_(nrs, this, t0, tf, step, o_y, userdata_, o_ydot);
  }
  else {
    // lag derivatives
    for (int s = nAB; s > 1; s--) {
      const auto Nbyte = (nDOFs_ * sizeof(dfloat)) * fieldOffset_;
      o_ydot.copyFrom(o_ydot, Nbyte, (s - 1) * Nbyte, (s - 2) * Nbyte);
    }

    userRHS_(nrs, this, t0, o_y, userdata_, o_ydot);

    nStagesSumManyKernel(nParticles_, fieldOffset_, nAB, nDOFs_, o_coeffAB, o_ydot, o_y);
  }

  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "integrate");
  }
}

namespace {
std::string lpm_vtu_data(std::string fieldName, int nComponent, int distance)
{
  return "<DataArray type=\"Float32\" Name=\"" + fieldName + "\" NumberOfComponents=\"" +
         std::to_string(nComponent) + "\" format=\"append\" offset=\"" + std::to_string(distance) + "\"/>\n";
}
} // namespace

std::set<std::string> lpm_t::nonCoordinateOutputDOFs() const
{
  std::set<std::string> outputDofFields;
  for (auto [dofName, outputDof] : outputDofs) {
    if (outputDof) {
      outputDofFields.insert(dofName);
    }
  }

  outputDofFields.erase("x");
  outputDofFields.erase("y");
  outputDofFields.erase("z");

  return outputDofFields;
}

int lpm_t::fieldOffset(int n) { return computeFieldOffset(n); }

void lpm_t::addParticles(int newNParticles, occa::memory o_yNewPart, occa::memory o_propNewPart)
{
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic(timerName + "addParticles", 1);
  }

  std::array<dlong, 2> counts = {newNParticles, size()};
  MPI_Allreduce(MPI_IN_PLACE, counts.data(), 2, MPI_DLONG, MPI_SUM, platform->comm.mpiComm);

  if (platform->comm.mpiRank == 0) {
    std::cout << "Adding " << counts[0] << " to " << counts[1] << " particles!\n";
  }

  int incomingOffset = computeFieldOffset(newNParticles);
  int newOffset = computeFieldOffset(this->nParticles_ + newNParticles);

  // check that the sizes of o_yNewPart, o_propNewPart are correct
  auto expectedYSize = incomingOffset * nDOFs_ * sizeof(dfloat);
  auto expectedPropSize = incomingOffset * nProps_ * sizeof(dfloat);
  if (o_yNewPart.size() != expectedYSize) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: o_yNewPart size is " << o_yNewPart.size() << " but expected " << expectedYSize
                << " bytes!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }

  if (o_propNewPart.size() != expectedPropSize) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "ERROR: o_propNewPart size is " << o_propNewPart.size() << " but expected "
                << expectedPropSize << " bytes!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }

  std::vector<dlong> remainingMap(this->nParticles_, 0);
  std::vector<dlong> insertMap(newNParticles, 0);

  // remainingMap[id] = id for existing particles
  std::iota(remainingMap.begin(), remainingMap.end(), 0);
  if (o_remainingMap.size() < this->nParticles_ * sizeof(dlong)) {
    if (o_remainingMap.size())
      o_remainingMap.free();
    o_remainingMap = platform->device.malloc(this->nParticles_ * sizeof(dlong));
  }
  o_remainingMap.copyFrom(remainingMap.data(), this->nParticles_ * sizeof(dlong));

  // insertMap[id] = id + nParticles_ for incoming particles
  std::iota(insertMap.begin(), insertMap.end(), this->nParticles_);
  if (o_insertMap.size() < newNParticles * sizeof(dlong)) {
    if (o_insertMap.size())
      o_insertMap.free();
    o_insertMap = platform->device.malloc(newNParticles * sizeof(dlong));
  }
  o_insertMap.copyFrom(insertMap.data(), newNParticles * sizeof(dlong));

  auto o_propOld = this->o_prop;
  auto o_interpFldOld = this->o_interpFld;
  auto o_yOld = this->o_y;
  auto o_ydotOld = this->o_ydot;

  if (nProps_) {
    o_prop = platform->device.malloc(newOffset * nProps_ * sizeof(dfloat));
  }
  if (nInterpFields_) {
    o_interpFld = platform->device.malloc(newOffset * nInterpFields_ * sizeof(dfloat));
  }
  o_y = platform->device.malloc(newOffset * nDOFs_ * sizeof(dfloat));
  o_ydot = platform->device.malloc(nAB * newOffset * nDOFs_ * sizeof(dfloat));

  // dummy arrays for remapParticlesKernel for new particles
  auto o_ydotDummy = platform->device.malloc(nAB * incomingOffset * nDOFs_ * sizeof(dfloat));
  occa::memory o_interpFldDummy;
  if (nInterpFields_) {
    o_interpFldDummy = platform->device.malloc(incomingOffset * nInterpFields_ * sizeof(dfloat));
  }

  // map existing particles to new data
  remapParticlesKernel(size(),
                       fieldOffset_,
                       newOffset,
                       nProps_,
                       nInterpFields_,
                       nDOFs_,
                       nAB,
                       o_remainingMap,
                       o_yOld,
                       o_ydotOld,
                       o_propOld,
                       o_interpFldOld,
                       o_y,
                       o_ydot,
                       o_prop,
                       o_interpFld);

  // map new particles to new data
  remapParticlesKernel(newNParticles,
                       incomingOffset,
                       newOffset,
                       nProps_,
                       nInterpFields_,
                       nDOFs_,
                       nAB,
                       o_insertMap,
                       o_yNewPart,
                       o_ydotDummy,
                       o_propNewPart,
                       o_interpFldDummy,
                       o_y,
                       o_ydot,
                       o_prop,
                       o_interpFld);

  o_propOld.free();
  o_interpFldOld.free();
  o_yOld.free();
  o_ydotOld.free();
  o_ydotDummy.free();
  o_interpFldDummy.free();

  nParticles_ += newNParticles;
  fieldOffset_ = newOffset;

  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "addParticles");
  }
}

void lpm_t::deleteParticles()
{
  int nDelete = 0;
  auto &code = interp->data().code;
  for (int pid = 0; pid < size(); ++pid) {
    if (code[pid] == findpts::CODE_NOT_FOUND) {
      ++nDelete;
    }
  }

  std::array<dlong, 2> counts = {nDelete, size()};
  MPI_Allreduce(MPI_IN_PLACE, counts.data(), 2, MPI_DLONG, MPI_SUM, platform->comm.mpiComm);

  const double deleteFraction = (double)counts[0] / (double)counts[1];

  // at least 50% of the particles must be deleted to justify resizing
  const double minDeleteFraction = 0.5;
  if (deleteFraction < minDeleteFraction) {
    return;
  }

  // apply tic here to get correct number of deletion events in timer output
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic(timerName + "deleteParticles", 1);
  }

  if (platform->comm.mpiRank == 0) {
    std::cout << "Deleting " << counts[0] << " of " << counts[1] << " particles!\n";
  }

  std::vector<dlong> remainingMap(size(), -1);

  dlong newId = 0;
  for (int pid = 0; pid < size(); ++pid) {
    if (code[pid] != findpts::CODE_NOT_FOUND) {
      remainingMap[pid] = newId;
      newId++;
    }
  }

  const auto Nbytes = size() * sizeof(dlong);
  if (o_remainingMap.size() < Nbytes) {
    if (o_remainingMap.size())
      o_remainingMap.free();
    o_remainingMap = platform->device.malloc(Nbytes);
  }
  o_remainingMap.copyFrom(remainingMap.data(), Nbytes);

  auto o_propOld = this->o_prop;
  auto o_interpFldOld = this->o_interpFld;
  auto o_yOld = this->o_y;
  auto o_ydotOld = this->o_ydot;

  const auto newNParticles = size() - nDelete;
  const auto newOffset = computeFieldOffset(newNParticles);

  if (nProps_) {
    o_prop = platform->device.malloc(newOffset * nProps_ * sizeof(dfloat));
  }
  if (nInterpFields_) {
    o_interpFld = platform->device.malloc(newOffset * nInterpFields_ * sizeof(dfloat));
  }
  o_y = platform->device.malloc(newOffset * nDOFs_ * sizeof(dfloat));
  o_ydot = platform->device.malloc(nAB * newOffset * nDOFs_ * sizeof(dfloat));

  remapParticlesKernel(size(),
                       fieldOffset_,
                       newOffset,
                       nProps_,
                       nInterpFields_,
                       nDOFs_,
                       nAB,
                       o_remainingMap,
                       o_yOld,
                       o_ydotOld,
                       o_propOld,
                       o_interpFldOld,
                       o_y,
                       o_ydot,
                       o_prop,
                       o_interpFld);

  o_propOld.free();
  o_interpFldOld.free();
  o_yOld.free();
  o_ydotOld.free();

  nParticles_ = newNParticles;
  fieldOffset_ = newOffset;

  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "deleteParticles");
  }
}

void lpm_t::writeFld(dfloat time)
{
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic(timerName + "write", 1);
  }

  static_assert(sizeof(float) == 4, "lpm_t::writeFld requires float be 32-bit");
  static_assert(sizeof(int) == 4, "lpm_t::writefld requires int be 32-bit");

  // Required to determine if points are outside of the domain
  // Do not output points outside of the domain

  dlong nPartOutput = 0;
  auto &code = interp->data().code;
  {
    auto o_xcoord = getDOF("x");
    auto o_ycoord = getDOF("y");
    auto o_zcoord = getDOF("z");
    interp->addPoints(size(), o_xcoord, o_ycoord, o_zcoord);

    // disable findpts kernel timer for this call
    auto saveLevel = getTimerLevel();
    setTimerLevel(TimerLevel::None);
    interp->find(false);
    setTimerLevel(saveLevel);

    for (int pid = 0; pid < size(); ++pid) {
      if (code[pid] != findpts::CODE_NOT_FOUND) {
        ++nPartOutput;
      }
    }
  }

  static int out_step = 0;
  ++out_step;

  MPI_Comm mpi_comm = platform->comm.mpiComm;
  int mpi_rank = platform->comm.mpiRank;
  int mpi_size = platform->comm.mpiCommSize;

  dlong globalNPartOutput = nPartOutput;

  MPI_Allreduce(MPI_IN_PLACE, &globalNPartOutput, 1, MPI_DLONG, MPI_SUM, platform->comm.mpiComm);

  if (globalNPartOutput == 0) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "No particles to output, skipping output step " << out_step << std::endl;
    }
    return;
  }

  std::ostringstream output;
  output << "par" << std::setw(5) << std::setfill('0') << out_step << ".vtu";
  std::string fname = output.str();

  dlong pOffset = 0;
  MPI_Exscan(&nPartOutput, &pOffset, 1, MPI_DLONG, MPI_SUM, mpi_comm);

  if (platform->comm.mpiRank == 0) {
    std::ofstream file(fname, std::ios::trunc);
    file.close();
  }

  MPI_File file_out;
  MPI_File_open(mpi_comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file_out);

  long long offset = 0;
  constexpr int dim = 3;

  // particles DOFs, sans coordinates
  auto particleOutputDOFs = nonCoordinateOutputDOFs();

  std::string message = "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
  message += "\t<UnstructuredGrid>\n";
  message += "\t\t<FieldData>\n";
  message += "\t\t\t<DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\"> " +
             std::to_string(time) + " </DataArray>\n";
  message += "\t\t\t<DataArray type=\"Int32\" Name=\"CYCLE\" NumberOfTuples=\"1\" format=\"ascii\"> " +
             std::to_string(out_step) + " </DataArray>\n";
  message += "\t\t</FieldData>\n";
  message += "\t\t<Piece NumberOfPoints=\"" + std::to_string(globalNPartOutput) + "\" NumberOfCells=\"0\">\n";
  message += "\t\t\t<Points>\n";
  message += "\t\t\t\t" + lpm_vtu_data("Position", dim, offset);
  offset += (dim * globalNPartOutput + 1) * sizeof(float);
  message += "\t\t\t</Points>\n";

  message += "\t\t\t<PointData>\n";

  // output particle DOFs
  for (auto &&dofName : particleOutputDOFs) {
    const auto Nfields = dofCounts.at(dofName);
    message += "\t\t\t\t" + lpm_vtu_data(dofName, Nfields, offset);
    offset += (Nfields * globalNPartOutput + 1) * sizeof(float);
  }

  // output particle properties
  for (auto [propName, isOutput] : outputProps) {
    if (!isOutput)
      continue;
    const auto Nfields = propCounts.at(propName);
    message += "\t\t\t\t" + lpm_vtu_data(propName, Nfields, offset);
    offset += (Nfields * globalNPartOutput + 1) * sizeof(float);
  }

  // output interpolated fields
  for (auto [interpFieldName, isOutput] : outputInterpFields) {
    if (!isOutput)
      continue;
    const auto Nfields = interpFieldCounts.at(interpFieldName);
    message += "\t\t\t\t" + lpm_vtu_data(interpFieldName, Nfields, offset);
    offset += (Nfields * globalNPartOutput + 1) * sizeof(float);
  }

  message += "\t\t\t</PointData>\n";

  message += "\t\t\t<Cells>\n";
  message += "\t\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\"/>\n";
  message += "\t\t\t\t<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\"/>\n";
  message += "\t\t\t\t<DataArray type=\"Int32\" Name=\"types\" format=\"ascii\"/>\n";
  message += "\t\t\t</Cells>\n";
  message += "\t\t</Piece>\n";
  message += "\t</UnstructuredGrid>\n";
  message += "\t<AppendedData encoding=\"raw\">\n";
  message += "_";

  if (platform->comm.mpiRank == 0) {
    MPI_File_write(file_out, message.c_str(), message.length(), MPI_CHAR, MPI_STATUS_IGNORE);
  }

  auto writeField = [&](int nFields, std::vector<float> &field) {
    MPI_Barrier(platform->comm.mpiComm);
    MPI_Offset position;
    MPI_File_get_size(file_out, &position);
    MPI_File_set_view(file_out, position, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
    if (platform->comm.mpiRank == 0) {
      int nbyte = nFields * globalNPartOutput * sizeof(float);
      MPI_File_write(file_out, &nbyte, 1, MPI_INT, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(platform->comm.mpiComm);
    MPI_File_get_size(file_out, &position);

    position += sizeof(float) * (nFields * pOffset);
    MPI_File_set_view(file_out, position, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
    MPI_File_write_all(file_out, field.data(), field.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
  };

  // output coordinates (required)
  {
    auto xCoord = getDOFHost("x");
    auto yCoord = getDOFHost("y");
    auto zCoord = getDOFHost("z");

    std::vector<float> positions(dim * nPartOutput, 0.0);
    dlong pid = 0;
    for (int particle = 0; particle < size(); ++particle) {
      if (code[particle] != findpts::CODE_NOT_FOUND) {
        positions[dim * pid + 0] = static_cast<float>(xCoord[particle]);
        positions[dim * pid + 1] = static_cast<float>(yCoord[particle]);
        positions[dim * pid + 2] = static_cast<float>(zCoord[particle]);
        pid++;
      }
    }

    writeField(dim, positions);
  }

  // other particle DOFs
  for (auto &&dofName : particleOutputDOFs) {
    auto dofHost = getDOFHost(dofName);
    auto Nfields = numDOFs(dofName);

    std::vector<float> dofFloat(Nfields * nPartOutput, 0.0);
    dlong pid = 0;
    for (int particle = 0; particle < size(); ++particle) {
      if (code[particle] != findpts::CODE_NOT_FOUND) {
        for (int fld = 0; fld < Nfields; ++fld) {
          dofFloat[Nfields * pid + fld] = static_cast<float>(dofHost[particle + fld * fieldOffset_]);
        }
        pid++;
      }
    }

    writeField(Nfields, dofFloat);
  }

  // particle properties
  for (auto [propName, isOutput] : outputProps) {
    if (!isOutput)
      continue;
    auto propHost = getPropHost(propName);
    auto Nfields = numProps(propName);

    std::vector<float> propFloat(Nfields * nPartOutput, 0.0);
    dlong pid = 0;
    for (int particle = 0; particle < size(); ++particle) {
      if (code[particle] != findpts::CODE_NOT_FOUND) {
        for (int fld = 0; fld < Nfields; ++fld) {
          propFloat[Nfields * pid + fld] = static_cast<float>(propHost[particle + fld * fieldOffset_]);
        }
        pid++;
      }
    }

    writeField(Nfields, propFloat);
  }

  // interpolated fields
  for (auto [interpFieldName, isOutput] : outputInterpFields) {
    if (!isOutput)
      continue;
    auto interpFieldHost = getInterpFieldHost(interpFieldName);
    auto Nfields = numFieldsInterp(interpFieldName);

    std::vector<float> interpFieldFloat(Nfields * nPartOutput, 0.0);
    dlong pid = 0;
    for (int particle = 0; particle < size(); ++particle) {
      if (code[particle] != findpts::CODE_NOT_FOUND) {
        for (int fld = 0; fld < Nfields; ++fld) {
          interpFieldFloat[Nfields * pid + fld] =
              static_cast<float>(interpFieldHost[particle + fld * fieldOffset_]);
        }
        pid++;
      }
    }

    writeField(Nfields, interpFieldFloat);
  }

  MPI_Barrier(platform->comm.mpiComm);
  MPI_Offset position;
  MPI_File_get_size(file_out, &position);
  MPI_File_set_view(file_out, position, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
  if (platform->comm.mpiRank == 0) {
    message = "";
    message += "</AppendedData>\n";
    message += "</VTKFile>";
    MPI_File_write(file_out, message.c_str(), message.length(), MPI_CHAR, MPI_STATUS_IGNORE);
  }

  MPI_File_close(&file_out);

  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "write");
  }
}

void lpm_t::registerKernels(occa::properties &kernelInfo)
{
  std::string installDir(getenv("NEKRS_HOME"));
  // build kernels
  std::string fileName, kernelName;
  const std::string suffix = "Hex3D";
  const std::string oklpath(getenv("NEKRS_KERNEL_DIR"));

  kernelName = "remapParticles";
  fileName = oklpath + "/plugins/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, kernelInfo);
}

void lpm_t::setTimerLevel(TimerLevel level)
{
  timerLevel = level;
  interp->setTimerLevel(level);
}

TimerLevel lpm_t::getTimerLevel() const { return timerLevel; }

void lpm_t::setTimerName(std::string name)
{
  timerName = name;
  interp->setTimerName(name);
}

namespace {
int parseNumParticles(std::string restartfile, const std::string &header)
{
  std::smatch npartmatch;
  bool found = std::regex_search(header, npartmatch, std::regex(R"(<Piece NumberOfPoints=\"(\d+)\")"));

  if (!found) {
    std::cout << "Could not read number of particles while reading " << restartfile << "!\n";
    return -1;
  }

  try {
    int nparticles = std::stoi(npartmatch[1].str());
    return nparticles;
  }
  catch (std::invalid_argument e) {
    std::cout << "Could not read number of particles while reading " << restartfile << "!\n";
    std::cout << "Exception said:\n" << e.what() << std::endl;
    return -1;
  }
}

auto parsePointData(std::string restartfile, std::string pointData)
{
  std::smatch match;
  bool found = std::regex_search(pointData, match, std::regex(R"(\s*Name=\"(.+?)\")"));
  if (!found) {
    std::cout << "Could not parse pointData while reading " << restartfile << "!\n";
    return std::make_tuple(std::string(), -1, -1);
  }
  auto fieldName = match[1].str();

  found = std::regex_search(pointData, match, std::regex(R"(\s*NumberOfComponents=\"(\d+)\")"));
  if (!found) {
    std::cout << "Could not parse " << fieldName << " number of components while reading " << restartfile
              << "!\n";
    return std::make_tuple(fieldName, -1, -1);
  }

  int numComponents = -1;
  try {
    numComponents = std::stoi(match[1].str());
  }
  catch (std::invalid_argument &e) {
    std::cout << "Could not parse " << fieldName << " number of components while reading " << restartfile
              << "!\n";
    std::cout << "Exception said:\n" << e.what() << std::endl;
    return std::make_tuple(fieldName, -1, -1);
  }

  int offset = -1;
  found = std::regex_search(pointData, match, std::regex(R"(\s*offset=\"(\d+)\")"));
  if (!found) {
    std::cout << "Could not parse " << fieldName << " offset while reading " << restartfile << "!\n";
    return std::make_tuple(fieldName, numComponents, -1);
  }

  try {
    offset = std::stoi(match[1].str());
  }
  catch (std::invalid_argument &e) {
    std::cout << "Could not parse " << fieldName << " offset while reading " << restartfile << "!\n";
    std::cout << "Exception said:\n" << e.what() << std::endl;
    return std::make_tuple(fieldName, numComponents, -1);
  }

  return std::make_tuple(fieldName, numComponents, offset);
}

auto readHeader(std::string restartFile)
{
  // read header of VTK UnstructuredGrid file format until reading after the <AppendedData encoding=\"raw\">
  // line
  std::string header;

  // associated with DataArray's inside <PointData> tag
  std::vector<std::string> pointData;

  // read metadata from restart file
  std::ifstream file(restartFile);
  std::string line;
  bool insidePointData = false;

  while (std::getline(file, line)) {
    header += line + "\n";
    if (line.find("<AppendedData encoding=\"raw\">") != std::string::npos) {
      break;
    }

    if (line.find("<PointData>") != std::string::npos) {
      insidePointData = true;
    }

    if (line.find("</PointData>") != std::string::npos) {
      insidePointData = false;
    }

    // gather PointData attributes to read later
    if (insidePointData && line.find("<DataArray") != std::string::npos) {
      pointData.push_back(line);
    }
  }
  file.close();

  header += "_";

  return std::make_tuple(header, pointData);
}

} // namespace

void lpm_t::restart(std::string restartFile)
{
  constexpr int dim = 3;
  auto [header, pointData] = readHeader(restartFile);

  // from header, extract number of particles stored in NumberOfPoints
  const auto nPartGlobal = parseNumParticles(restartFile, header);
  auto nPartLocal = nPartGlobal / platform->comm.mpiCommSize;
  // distribute remaining particles
  const int remainder = nPartGlobal % platform->comm.mpiCommSize;
  if (platform->comm.mpiRank < remainder) {
    nPartLocal++;
  }

  this->construct(nPartLocal);

  dlong pOffset = 0;
  MPI_Exscan(&nPartLocal, &pOffset, 1, MPI_DLONG, MPI_SUM, platform->comm.mpiComm);

  std::map<std::string, std::tuple<int, int>> fieldToInfo;
  for (auto &&field : pointData) {
    auto [fieldName, numComponents, offset] = parsePointData(restartFile, field);
    fieldToInfo[fieldName] = std::make_tuple(numComponents, offset);
  }

  // start by reading coordinates, starting at the position left by the header
  MPI_File file_in;
  MPI_File_open(platform->comm.mpiComm, restartFile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file_in);

  MPI_Offset position = header.length();
  MPI_File_set_view(file_in, position, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

  // first field is number of bytes in coordinate data
  int nPointData = 0;
  MPI_File_read(file_in, &nPointData, 1, MPI_INT, MPI_STATUS_IGNORE);
  nPointData /= dim;
  nPointData /= sizeof(float);

  if (nPointData != nPartGlobal) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "Number of particles in header (" << nPartGlobal
                << ") does not match number of particles in file (" << nPointData << ")!\n";
    }
    nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
  }

  position = header.length();
  position += sizeof(float) * (dim * pOffset + 1);
  MPI_File_set_view(file_in, position, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

  std::vector<float> coords(nPartLocal * dim);

  std::vector<double> xCoord(nPartLocal);
  std::vector<double> yCoord(nPartLocal);
  std::vector<double> zCoord(nPartLocal);

  MPI_File_read(file_in, coords.data(), nPartLocal * dim, MPI_FLOAT, MPI_STATUS_IGNORE);
  for (int pid = 0; pid < nPartLocal; ++pid) {
    xCoord[pid] = static_cast<dfloat>(coords[dim * pid + 0]);
    yCoord[pid] = static_cast<dfloat>(coords[dim * pid + 1]);
    zCoord[pid] = static_cast<dfloat>(coords[dim * pid + 2]);
  }

  auto o_xCoord = getDOF("x");
  auto o_yCoord = getDOF("y");
  auto o_zCoord = getDOF("z");

  o_xCoord.copyFrom(xCoord.data(), nPartLocal * sizeof(dfloat));
  o_yCoord.copyFrom(yCoord.data(), nPartLocal * sizeof(dfloat));
  o_zCoord.copyFrom(zCoord.data(), nPartLocal * sizeof(dfloat));

  auto readField = [&](std::string fieldName, int expectedNumComponents, int offset) {
    if (fieldType.count(fieldName) == 0) {
      if (platform->comm.mpiRank == 0) {
        std::cout << "Encountered unregistered field " << fieldName << " while reading restart "
                  << restartFile << "!\n";
      }
      nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
    }

    auto type = fieldType[fieldName];

    int nComponents = -1;
    occa::memory o_fld;
    if (type == FieldType::DOF) {
      nComponents = dofCounts.at(fieldName);
      o_fld = getDOF(fieldName);
    }
    else if (type == FieldType::PROP) {
      nComponents = propCounts.at(fieldName);
      o_fld = getProp(fieldName);
    }
    else if (type == FieldType::INTERP_FIELD) {
      nComponents = interpFieldCounts.at(fieldName);
      o_fld = getInterpField(fieldName);
    }

    if (nComponents != expectedNumComponents) {
      if (platform->comm.mpiRank == 0) {
        std::cout << "Excepted number of components for field " << fieldName << " (" << expectedNumComponents
                  << ") does not match number of components (" << nComponents << ") in restart file "
                  << restartFile << "!\n";
      }
      nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
    }

    position = header.length();
    position += offset;

    MPI_File_set_view(file_in, position, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

    // first field is number of bytes in coordinate data
    int nPointData = 0;
    MPI_File_read(file_in, &nPointData, 1, MPI_INT, MPI_STATUS_IGNORE);
    nPointData /= nComponents;
    nPointData /= sizeof(float);

    if (nPointData != nPartGlobal) {
      if (platform->comm.mpiRank == 0) {
        std::cout << "Number of particles in header (" << nPartGlobal
                  << ") does not match number of particles in file (" << nPointData << ") when reading field "
                  << fieldName << "!\n";
      }
      nrsCheck(1, platform->comm.mpiComm, EXIT_FAILURE, "", "");
    }

    std::vector<float> fld(nPartLocal * nComponents);
    std::vector<dfloat> fldHost(this->fieldOffset() * nComponents);

    position = header.length();
    position += offset;
    position += sizeof(float) * (nComponents * pOffset + 1);

    MPI_File_set_view(file_in, position, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

    MPI_File_read(file_in, fld.data(), nPartLocal * nComponents, MPI_FLOAT, MPI_STATUS_IGNORE);
    for (int pid = 0; pid < nPartLocal; ++pid) {
      for (int component = 0; component < nComponents; ++component) {
        fldHost[pid + component * this->fieldOffset()] =
            static_cast<dfloat>(fld[nComponents * pid + component]);
      }
    }

    o_fld.copyFrom(fldHost.data(), this->fieldOffset() * nComponents * sizeof(dfloat));
  };

  for (auto &&[fieldName, info] : fieldToInfo) {
    auto [numComponents, offset] = info;
    readField(fieldName, numComponents, offset);
  }

  MPI_File_close(&file_in);
}
