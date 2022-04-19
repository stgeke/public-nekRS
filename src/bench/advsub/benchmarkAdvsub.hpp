#include "occa.hpp"

template <typename T>
occa::kernel benchmarkAdvsub(int Nfields,
                             int Nelements,
                             int Nq,
                             int cubNq,
                             int nEXT,
                             bool dealias,
                             int verbosity,
                             T NtestsOrTargetTime,
                             bool requiresBenchmark);