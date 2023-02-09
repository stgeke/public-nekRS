#if !defined(nekrs_lpm_hpp_)
#define nekrs_lpm_hpp_

#include "nrssys.hpp"
#include <functional>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include "pointInterpolation.hpp"

class nrs_t;

// Lagrangian particle manager
class lpm_t {
public:
  // User-defined RHS function
  // INPUTS:
  //   nrs: NekRS object
  //   lpm: particle manager
  //   time: current time
  //   o_y: particle degrees of freedom
  //        DO NOT ALTER!
  //   userdata: user-defined data
  // OUTPUTS:
  //   o_ydot: derivatives
  using rhsFunc_t = std::function<
      void(nrs_t *nrs, lpm_t *lpm, dfloat time, occa::memory o_y, void *userdata, occa::memory o_ydot)>;

  // User-defined ODE solver
  // Integrate from t0 to tf
  // INPUTS:
  //   nrs: NekRS object
  //   lpm: particle manager
  //   time: current time
  //   dt: time step(s)
  //   step: time step number
  //   o_y: particle degrees of freedom
  //   userdata: user-defined data
  //   o_ydot: scratch space
  using odeSolverFunc_t = std::function<void(nrs_t *nrs,
                                             lpm_t *lpm,
                                             dfloat t0,
                                             dfloat tf,
                                             int step,
                                             occa::memory o_y,
                                             void *userdata,
                                             occa::memory o_ydot)>;

  // By default, nAB is set to nrs->nEXT
  lpm_t(nrs_t *nrs, dfloat newton_tol_ = 0.0);

  // nAB: maximum AB order (1,2, or 3)
  //   nAB must be less than or equal to nrs->nEXT
  lpm_t(nrs_t *nrs, int nAB, dfloat newton_tol_ = 0.0);

  ~lpm_t() = default;

  lpm_t(const lpm_t &) = delete;
  lpm_t(const lpm_t &&) = delete;

  // static kernel registration function
  // This *MUST* be called during UDF_LoadKernels
  static void registerKernels(occa::properties &kernelInfo);

  // Register a degree of freedom (case insensitive)
  // By default, the particle coordinates are already registered
  // An optional bool argument may be used to flag a field as not needing to be output
  // during a lpm_t::writeFld(...) call.
  // By default, only the particle coordinates are automatically registered.
  // Pre:
  //   constructed() = false
  void registerDOF(std::string dofName, bool output = true);

  // Multi-component field version
  // On output, this field will be output to the VTU file as a vector quantity.
  void registerDOF(dlong Nfields, std::string dofName, bool output = true);

  // Properties associated with the particle
  // NOTE: these are _not_ degrees of freedom. Use registerDOF(...) for that.
  // As registerDOF(...), there's an optional bool output argument to
  // flag whether a field should be output during a lpm_t::writeFld(...) call.
  // Pre:
  //   constructed() = false
  void registerProp(std::string propName, bool output = true);

  // Multi-component field version
  // On output, this field will be output to the VTU file as a vector quantity.
  void registerProp(dlong Nfields, std::string propName, bool output = true);

  // Fields associated with the fluid mesh to be interpolated
  // to the particle locations.
  // On input, o_fld is a field on the fluid mesh to be interpolated.
  // NOTE: these are _not_ degrees of freedom. Use registerDOF(...) for that.
  // As registerDOF(...), there's an optional bool output argument to
  // flag whether a field should be output during a lpm_t::writeFld(...) call.
  // Pre:
  //   constructed() = false
  void registerInterpField(std::string interpFieldName, occa::memory o_fld, bool output = true);

  // Multi-component field version
  // Prefer using this version for multi-component fields, as the performance
  // will be better during interpolation.
  // On output, this field will be output to the VTU file as a vector quantity.
  void
  registerInterpField(std::string interpFieldName, dlong Nfields, occa::memory o_fld, bool output = true);

  // Get field index associated with a degree of freedom
  int dofId(std::string dofName) const;

  // Number of fields associated with a DOF
  int numDOFs(std::string dofName) const;

  // Get field index associated with a property
  int propId(std::string propName) const;

  // Number of fields associated with a property
  int numProps(std::string propName) const;

  // Get field index associated with an interpolated field
  int interpFieldId(std::string interpFieldName) const;

  // Number of fields associated with an interpolated field
  int numFieldsInterp(std::string interpFieldName) const;

  bool constructed() const { return constructed_; }

  // Required user RHS
  void setUserRHS(rhsFunc_t userRHS);

  // Optionally set user ODE solver
  void setUserODESolver(odeSolverFunc_t userODESolver) { userODESolver_ = userODESolver; };

  // Add optional userdata ptr to be passed to userRHS
  void addUserData(void *userdata);

  // Construct particles on manager
  void construct(int nParticles);

  // Page-aligned offset >= nParticles
  // Required to access particle-specific fields
  // NOT valid until construct() is called
  int fieldOffset() const { return fieldOffset_; }

  // Compute page-aligned offset >= n
  static int fieldOffset(int n);

  // Number of particle degrees of freedom
  int nDOFs() const { return nDOFs_; }

  // Number of particle properties
  int nProps() const { return nProps_; }

  // Number of interpolated fields
  int nInterpFields() const { return nInterpFields_; }

  // Number of particles
  int size() const { return nParticles_; }

  // Integrate from time t0 to tf
  // Pre:
  //   constructed() = true
  void integrate(dfloat t0, dfloat tf, int step);

  // Write particle data to file
  void writeFld(dfloat time);

  // Read particle data from file
  // Can be called in lieu of construct
  void restart(std::string restartFile);

  // Get particle degrees of freedom on device
  // Pre:
  //  constructed() = true
  occa::memory getDOF(int dofId);
  occa::memory getDOF(std::string dofName);

  // Get particle coordinates on host
  std::vector<dfloat> getDOFHost(std::string dofName);

  // Get particle property on device
  // Pre:
  //  constructed() = true
  occa::memory getProp(int propId);
  occa::memory getProp(std::string propName);

  // Get particle properties on host
  std::vector<dfloat> getPropHost(std::string propName);

  // Get interpolated field on device
  // Pre:
  //  constructed() = true
  occa::memory getInterpField(int interpFieldId);
  occa::memory getInterpField(std::string interpFieldName);

  // Get interpolated fields on host
  std::vector<dfloat> getInterpFieldHost(std::string interpFieldName);

  // Get the underlying pointInterpolation_t object
  pointInterpolation_t &interpolator() { return *interp; }

  // Interpolate all interpolated fields from the fluid mesh to the particle locations
  void interpolate();

  // Interpolate specific field name from the fluid mesh to the particle locations
  void interpolate(std::string interpFieldName);

  // Add new particles
  // NOTE: o_y, o_prop are assumed to be page-aligned, and
  // must contain all of the fields associated with the lpm_t object.
  // The user is responsible for deleting o_yNewPart and o_propNewPart, if needed.
  // The lagged ydot values are assumed to be zero.
  void addParticles(int nParticles, occa::memory o_yNewPart, occa::memory o_propNewPart);

  occa::memory o_prop;      // particle properties
  occa::memory o_interpFld; // interpolated field outputs
  occa::memory o_y;         // particle degrees of freedom
  occa::memory o_ydot;      // derivatives

  // set timer level
  void setTimerLevel(TimerLevel level);
  TimerLevel getTimerLevel() const;

  // set timer name
  // this is used to prefix the timer names
  void setTimerName(std::string name);

private:
  // delete particles that have left the domain
  void deleteParticles();

  // generate set of all output DOFs, sans {x,y,z}
  std::set<std::string> nonCoordinateOutputDOFs() const;

  void coeff(dfloat *dt, int tstep);

  std::string timerName = "lpm_t::";
  TimerLevel timerLevel = TimerLevel::Basic;
  nrs_t *nrs = nullptr;
  int nAB;
  dfloat newton_tol;
  std::unique_ptr<pointInterpolation_t> interp;

  int nParticles_ = 0;
  int nDOFs_ = 0;
  int nProps_ = 0;
  int nInterpFields_ = 0;
  int fieldOffset_ = 0; // page-aligned offset >= nParticles
  bool constructed_ = false;

  std::vector<dfloat> coeffAB;
  occa::memory o_coeffAB; // AB coefficients

  rhsFunc_t userRHS_ = nullptr;
  odeSolverFunc_t userODESolver_ = nullptr;

  enum class FieldType { DOF, PROP, INTERP_FIELD };

  // Map between dof/prop/interpField names and which type of field they are
  // This is especially useful when reading in restart files
  std::map<std::string, FieldType> fieldType;

  // DOFs
  std::map<std::string, int> dofIds;
  std::map<std::string, int> dofCounts;
  std::map<std::string, bool> outputDofs;

  // Properties
  std::map<std::string, int> propIds;
  std::map<std::string, int> propCounts;
  std::map<std::string, bool> outputProps;

  // Interpolated fields
  std::map<std::string, int> interpFieldIds;
  std::map<std::string, int> interpFieldCounts;
  std::map<std::string, bool> outputInterpFields;
  std::map<std::string, occa::memory> interpFieldInputs;

  // map from current particle id to new particle id when adding/deleting particles
  occa::memory o_remainingMap;

  // map new particles to particle id when adding particles
  occa::memory o_insertMap;

  void *userdata_ = nullptr;
  occa::kernel nStagesSumManyKernel;
  occa::kernel remapParticlesKernel;
};

#endif
