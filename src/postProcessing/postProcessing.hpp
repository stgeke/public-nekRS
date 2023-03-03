#if !defined(nekrs_post_hpp_)
#define nekrs_post_hpp_

#include "nrs.hpp"

namespace postProcessing
{
void planarAvg(nrs_t *nrs, const std::string& dir, int NELGX, int NELGY, int NELGZ, int nflds, occa::memory o_avg);
dfloat viscousDrag(nrs_t *nrs, int nbID, const occa::memory& o_bID, occa::memory& o_Sij);
void strainRate(nrs_t *nrs, bool smooth, occa::memory& o_Sij);
}

#endif
