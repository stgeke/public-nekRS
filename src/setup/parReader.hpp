#if !defined(nekrs_parreader_hpp_)
#define nekrs_parreader_hpp_

#include "nrs.hpp"
#include "inipp.hpp"

void parRead(inipp::Ini *par, std::string setupFile, MPI_Comm comm, setupAide &options);

#endif
