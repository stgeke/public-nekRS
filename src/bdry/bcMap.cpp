#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <set>

#include "nrs.hpp"
#include "platform.hpp"
#include "udf.hpp"

#include <elliptic.h>
#include "alignment.hpp"
#include "bcMap.hpp"

namespace {
boundaryAlignment_t computeAlignment(mesh_t *mesh, dlong element, dlong face)
{
  const dfloat alignmentTol = 1e-3;
  dfloat nxDiff = 0.0;
  dfloat nyDiff = 0.0;
  dfloat nzDiff = 0.0;

  for (int fp = 0; fp < mesh->Nfp; ++fp) {
    const dlong sid = mesh->Nsgeo * (mesh->Nfaces * mesh->Nfp * element + mesh->Nfp * face + fp);
    const dfloat nx = mesh->sgeo[sid + NXID];
    const dfloat ny = mesh->sgeo[sid + NYID];
    const dfloat nz = mesh->sgeo[sid + NZID];
    nxDiff += std::abs(std::abs(nx) - 1.0);
    nyDiff += std::abs(std::abs(ny) - 1.0);
    nzDiff += std::abs(std::abs(nz) - 1.0);
  }

  nxDiff /= mesh->Nfp;
  nyDiff /= mesh->Nfp;
  nzDiff /= mesh->Nfp;

  if (nxDiff < alignmentTol)
    return boundaryAlignment_t::X;
  if (nyDiff < alignmentTol)
    return boundaryAlignment_t::Y;
  if (nzDiff < alignmentTol)
    return boundaryAlignment_t::Z;

  return boundaryAlignment_t::UNALIGNED;
}
} // namespace

static bool meshConditionsDerived = false;
static std::set<std::string> fields;
// stores for every (field, boundaryID) pair a bcID
static std::map<std::pair<std::string, int>, int> bToBc;
static int nbid[] = {-1, -1};
static bool importFromNek = true;

static std::map<std::string, int> vBcTextToID = {
    {"periodic", 0},
    {"zerovalue", bcMap::bcTypeW},
    {"interpolation", bcMap::bcTypeINT},
    {"codedfixedvalue", bcMap::bcTypeV},
    {"zeroxvalue/zerogradient", bcMap::bcTypeSYMX},
    {"zeroyvalue/zerogradient", bcMap::bcTypeSYMY},
    {"zerozvalue/zerogradient", bcMap::bcTypeSYMZ},
    {"zeronvalue/zerogradient", bcMap::bcTypeSYM},
    {"zeroxvalue/codedfixedgradient", bcMap::bcTypeSHLX},
    {"zeroyvalue/codedfixedgradient", bcMap::bcTypeSHLY},
    {"zerozvalue/codedfixedgradient", bcMap::bcTypeSHLZ},
    {"zeronvalue/codedfixedgradient", bcMap::bcTypeSHL},
    {"zeroyzvalue/fixedgradient", bcMap::bcTypeONX},
    {"zeroxzvalue/fixedgradient", bcMap::bcTypeONY},
    {"zeroxyvalue/fixedgradient", bcMap::bcTypeONZ},
    // {"zeroTValue/fixedgradient", bcMap::bcTypeON},
    {"fixedgradient", bcMap::bcTypeO},
    {"zerogradient", bcMap::bcTypeO},
    {"none", bcMap::bcTypeNone}
};

static std::map<int, std::string> vBcIDToText = {
    {0, "periodic"},
    {bcMap::bcTypeW, "zeroValue"},
    {bcMap::bcTypeINT, "interpolation"},
    {bcMap::bcTypeV, "codedFixedValue"},
    {bcMap::bcTypeSYMX, "zeroXValue/zeroGradient"},
    {bcMap::bcTypeSYMY, "zeroYValue/zeroGradient"},
    {bcMap::bcTypeSYMZ, "zeroZValue/zeroGradient"},
    {bcMap::bcTypeSYM, "zeroNValue/zeroGradient"},
    {bcMap::bcTypeSHLX, "zeroXValue/codedFixedGradient"},
    {bcMap::bcTypeSHLY, "zeroYValue/codedFixedGradient"},
    {bcMap::bcTypeSHLZ, "zeroZValue/codedFixedGradient"},
    {bcMap::bcTypeSHL, "zeroNValue/codedFixedGradient"},
    {bcMap::bcTypeONX, "zeroYZValue/fixedGradient"},
    {bcMap::bcTypeONY, "zeroXZValue/fixedGradient"},
    {bcMap::bcTypeONZ, "zeroXYValue/fixedGradient"},
    // {bcMap::bcTypeON, "zeroTValue/fixedGradient"},
    {bcMap::bcTypeO, "fixedGradient"},
    {bcMap::bcTypeNone ,"none"}
};

static std::map<std::string, int> sBcTextToID = {{"periodic", 0},
                                                 {"interpolation", bcMap::bcTypeINTS},
                                                 {"codedfixedvalue", bcMap::bcTypeS},
                                                 {"zerogradient", bcMap::bcTypeF0},
                                                 {"codedfixedgradient", bcMap::bcTypeF},
                                                 {"codedFixedgradient", bcMap::bcTypeF},
                                                 {"none", bcMap::bcTypeNone}
};

static std::map<int, std::string> sBcIDToText = {{0, "periodic"},
                                                 {bcMap::bcTypeINTS, "interpolation"},
                                                 {bcMap::bcTypeS, "codedFixedValue"},
                                                 {bcMap::bcTypeF0, "zeroGradient"},
                                                 {bcMap::bcTypeF, "codedFixedGradient"},
                                                 {bcMap::bcTypeNone ,"none"}
};

static void v_setup(std::string s);
static void s_setup(std::string s);

static void v_setup(std::string field, std::vector<std::string> slist)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // platform not available yet

  int foundAligned = 0;
  int foundUnaligned = 0;

  for (int bid = 0; bid < slist.size(); bid++) {
    std::string key = slist[bid];

    if (key.compare("p") == 0) key = "periodic";

    if (key.compare("w") == 0) key = "zerovalue";
    if (key.compare("wall") == 0) key = "zerovalue";

    if (key.compare("int") == 0)
      key = "interpolation";
    if (key.compare("interpolation") == 0)
      key = "interpolation";

    if (key.compare("inlet") == 0) key = "codedfixedvalue";
    if (key.compare("v") == 0) key = "codedfixedvalue";

    if (key.compare("mv") == 0) key = "codedfixedvalue";
    if (key.compare("codedfixedvalue+moving") == 0) key = "codedfixedvalue";

    if (key.compare("slipx") == 0 || key.compare("symx")  == 0) {
      key = "zeroxvalue/zerogradient";
      foundAligned++;
    }
    if (key.compare("slipy") == 0 || key.compare("symy")  == 0) {
      key = "zeroyvalue/zerogradient";
      foundAligned++;
    }
    if (key.compare("slipz") == 0 || key.compare("symz")  == 0) {
      key = "zerozvalue/zerogradient";
      foundAligned++;
    }
    if (key.compare("sym") == 0) {
      key = "zeronvalue/zerogradient";
      foundUnaligned++;
    }

    if (key.compare("tractionx") == 0 || key.compare("shlx") == 0) {
      key = "zeroxvalue/codedfixedgradient";
      foundAligned++;
    }
    if (key.compare("tractiony") == 0 || key.compare("shly") == 0) {
      key = "zeroyvalue/codedfixedgradient";
      foundAligned++;
    }
    if (key.compare("tractionz") == 0 || key.compare("shlz") == 0) {
      key = "zerozvalue/codedfixedgradient";
      foundAligned++;
    }
    if (key.compare("shl") == 0) {
      key = "zeronvalue/codedfixedgradient";
      foundUnaligned++;
    }

    if (key.compare("outlet") == 0) key = "fixedgradient";
    if (key.compare("outflow") == 0) key = "fixedgradient";
    if (key.compare("o") == 0) key = "fixedgradient";

    if (key.compare("onx") == 0) {
      key = "zeroyzvalue/fixedgradient";
      foundAligned++;
    }
    if (key.compare("ony") == 0) {
      key = "zeroxzvalue/fixedgradient";
      foundAligned++;
    }
    if (key.compare("onz") == 0) {
      key = "zeroxyvalue/fixedgradient";
      foundAligned++;
    }
// not supported yet
#if 0
    if (key.compare("on") == 0) {
      key = "zerotvalue/fixedgradient";
      foundUnaligned++;
    }
#endif

    nrsCheck(vBcTextToID.find(key) == vBcTextToID.end(), MPI_COMM_WORLD, EXIT_FAILURE,
             "Invalid velocity bcType (%s)\n", key.c_str());

    bToBc[make_pair(field, bid)] = vBcTextToID.at(key);
  }

  nrsCheck(foundAligned && foundUnaligned, MPI_COMM_WORLD, EXIT_FAILURE,
           "Aligned together with unaligned mixed boundary types are not supported!\n", "");
}

static void s_setup(std::string field, std::vector<std::string> slist)
{
  for (int bid = 0; bid < slist.size(); bid++) {
    std::string key = slist[bid];
    if (key.compare("p") == 0)
      key = "periodic";

    if (key.compare("int") == 0)
      key = "interpolation";
    if (key.compare("interpolation") == 0)
      key = "interpolation";

    if (key.compare("t") == 0)
      key = "codedfixedvalue";
    if (key.compare("inlet") == 0)
      key = "codedfixedvalue";

    if (key.compare("flux") == 0)
      key = "codedfixedgradient";
    if (key.compare("f") == 0)
      key = "codedfixedgradient";

    if (key.compare("zeroflux") == 0)
      key = "zerogradient";
    if (key.compare("i") == 0)
      key = "zerogradient";
    if (key.compare("insulated") == 0)
      key = "zerogradient";

    if (key.compare("outflow") == 0)
      key = "zerogradient";
    if (key.compare("outlet") == 0)
      key = "zerogradient";
    if (key.compare("o") == 0)
      key = "zerogradient";

    nrsCheck(sBcTextToID.find(key) == sBcTextToID.end(), MPI_COMM_WORLD, EXIT_FAILURE,
             "Invalid scalar bcType (%s)\n", key.c_str());

    bToBc[make_pair(field, bid)] = sBcTextToID.at(key);
  }
}

namespace bcMap
{
bool useNekBCs() { return importFromNek; }

void setup(std::vector<std::string> slist, std::string field)
{
  if (slist.size() == 0)
    return;

  importFromNek = false;

  if (slist[0].compare("none") == 0)
    return;

  fields.insert(field);

  if (field.compare(0, 8, "scalar00") == 0) /* tmesh */ 
    nbid[1] = slist.size();
  else 
    nbid[0] = slist.size();

  if (field.compare("velocity") == 0)
    v_setup(field, slist);
  else if (field.compare("mesh") == 0)
    v_setup(field, slist);
  else if (field.compare(0, 6, "scalar") == 0)
    s_setup(field, slist);
}

void deriveMeshBoundaryConditions(std::vector<std::string> velocityBCs)
{
  if (velocityBCs.size() == 0 || velocityBCs[0].compare("none") == 0) return;

  meshConditionsDerived = true;

  const std::string field = "mesh";

  fields.insert(field);

  for (int bid = 0; bid < velocityBCs.size(); bid++) {
    const std::string keyIn = velocityBCs[bid];

    std::string key = "zeronvalue/zerogradient"; // default

    if (keyIn.compare("none") == 0) key = "none";

    if (keyIn.compare("zerovalue") == 0) key = "zerovalue";
    if (keyIn.compare("codedfixedvalue") == 0) key = "codedfixedvalue";

    if (keyIn.compare("p") == 0) key = "periodic";

    if (keyIn.compare("w") == 0) key = "zerovalue";
    if (keyIn.compare("wall") == 0) key = "zerovalue";
    if (keyIn.compare("inlet") == 0) key = "zerovalue";
    if (keyIn.compare("v") == 0) key = "zerovalue";

    if (key.compare("int") == 0) key = "zerovalue";
    if (key.compare("interpolation") == 0) key = "zerovalue";

    if (keyIn.compare("mv") == 0) key = "codedfixedvalue";
    if (keyIn.compare("codedfixedvalue+moving") == 0) key = "codedfixedvalue";

    nrsCheck(vBcTextToID.find(key) == vBcTextToID.end(), MPI_COMM_WORLD, EXIT_FAILURE,
             "Invalid bcType (%s)\n", key.c_str());

    bToBc[make_pair(field, bid)] = vBcTextToID.at(key);
  }
}

// return boundary type id for a given boundary id
int id(int bid, std::string field)
{
  if (bid < 1)
    return NO_OP;

  try {
    return bToBc.at({field, bid - 1});
  }
  catch (const std::out_of_range &oor) {
    nrsAbort(MPI_COMM_SELF, EXIT_FAILURE, 
             "bcMap::id bid:%d field:%s lookup failed!\n", bid, field.c_str());
  }

  return -1;
}

int ellipticType(int bid, std::string field)
{
  if (bid < 1)
    return NO_OP;

  try {
    int bcType = -1;
    if (field.compare("x-velocity") == 0 || field.compare("x-mesh") == 0) {
      const std::string fld = (field.compare("x-velocity") == 0) ? "velocity" : "mesh";
      const int bcID = bToBc.at({fld, bid - 1}); 
      
      bcType = DIRICHLET;
      if (bcID == bcTypeO)
        bcType = NEUMANN;
      if (bcID == bcTypeSYMY || bcID == bcTypeSHLY)
        bcType = NEUMANN;
      if (bcID == bcTypeSYMZ || bcID == bcTypeSHLZ)
        bcType = NEUMANN;
      if (bcID == bcTypeSYM || bcID == bcTypeSHL)
        bcType = ZERO_NORMAL;
      if (bcID == bcTypeONX)
        bcType = NEUMANN;
      if (bcID == bcTypeON)
        bcType = ZERO_TANGENTIAL;
      if (bcID == bcTypeNone)
        bcType = NO_OP;
    }
    else if (field.compare("y-velocity") == 0 || field.compare("y-mesh") == 0) {
      const std::string fld = (field.compare("y-velocity") == 0) ? "velocity" : "mesh";
      const int bcID = bToBc.at({fld, bid - 1}); 

      bcType = DIRICHLET;
      if (bcID == bcTypeO)
        bcType = NEUMANN;
      if (bcID == bcTypeSYMX || bcID == bcTypeSHLX)
        bcType = NEUMANN;
      if (bcID == bcTypeSYMZ || bcID == bcTypeSHLZ)
        bcType = NEUMANN;
      if (bcID == bcTypeSYM || bcID == bcTypeSHL)
        bcType = ZERO_NORMAL;
      if (bcID == bcTypeONY)
        bcType = NEUMANN;
      if (bcID == bcTypeON)
        bcType = ZERO_TANGENTIAL;
      if (bcID == bcTypeNone)
        bcType = NO_OP;
    }
    else if (field.compare("z-velocity") == 0 || field.compare("z-mesh") == 0) {
      const std::string fld = (field.compare("z-velocity") == 0) ? "velocity" : "mesh";
      const int bcID = bToBc.at({fld, bid - 1}); 

      bcType = DIRICHLET;
      if (bcID == bcTypeO)
        bcType = NEUMANN;
      if (bcID == bcTypeSYMX || bcID == bcTypeSHLX)
        bcType = NEUMANN;
      if (bcID == bcTypeSYMY || bcID == bcTypeSHLY)
        bcType = NEUMANN;
      if (bcID == bcTypeSYM || bcID == bcTypeSHL)
        bcType = ZERO_NORMAL;
      if (bcID == bcTypeONZ)
        bcType = NEUMANN;
      if (bcID == bcTypeON)
        bcType = ZERO_TANGENTIAL;
      if (bcID == bcTypeNone)
        bcType = NO_OP;
    }
    else if (field.compare("pressure") == 0) {
      const int bcID = bToBc.at({"velocity", bid - 1});
      bcType = NEUMANN;
      if (bcID == bcTypeO || bcID == bcTypeONX || bcID == bcTypeONY || bcID == bcTypeONZ || bcID == bcTypeON)
        bcType = DIRICHLET;
      if (bcID == bcTypeNone)
        bcType = NO_OP;
    }
    else if (field.compare(0, 6, "scalar") == 0) {
      const int bcID = bToBc.at({field, bid - 1});

      bcType = NEUMANN;
      if (bcID == bcTypeS)
        bcType = DIRICHLET;
      if (bcID == bcTypeNone)
        bcType = NO_OP;
    }

    nrsCheck(bcType == -1, MPI_COMM_SELF, EXIT_FAILURE,
             "ellipticType bid:%d field:%s lookup failed!\n", bid, field.c_str());

    return bcType;
  }
  catch (const std::out_of_range &oor) {
    nrsAbort(MPI_COMM_SELF, EXIT_FAILURE,
             "ellipticType bid:%d field:%s lookup failed!\n", bid, field.c_str());
  }

  return 0;
}

std::string text(int bid, std::string field)
{
  if (bid < 1) return std::string();

  const int bcID = bToBc.at({field, bid - 1});

  if (bcID == bcTypeNone) 
    return std::string("");

  if (field.compare("velocity") == 0 && (bcID == bcTypeV || bcID == bcTypeINT))
    oudfFindDirichlet(field);
  if (field.compare("mesh") == 0 && bcID == bcTypeV)
    oudfFindDirichlet(field);
  if (field.compare("pressure") == 0 &&
      (bcID == bcTypeONX || bcID == bcTypeONY || bcID == bcTypeONZ || bcID == bcTypeON || bcID == bcTypeO))
    oudfFindDirichlet(field);
  if (field.compare(0, 6, "scalar") == 0 && (bcID == bcTypeS || bcID == bcTypeINTS))
    oudfFindDirichlet(field);

  if (field.compare("velocity") == 0 &&
      (bcID == bcTypeSHLX || bcID == bcTypeSHLY || bcID == bcTypeSHLZ || bcID == bcTypeSHL))
    oudfFindNeumann(field);
  if (field.compare("mesh") == 0 && bcID == bcTypeSHL)
    oudfFindNeumann(field);
  if (field.compare(0, 6, "scalar") == 0 && bcID == bcTypeF)
    oudfFindNeumann(field);

  if (field.compare("velocity") == 0 || field.compare("mesh") == 0)
    return vBcIDToText.at(bcID);
  else if (field.compare(0, 6, "scalar") == 0)
    return sBcIDToText.at(bcID);

  nrsAbort(MPI_COMM_SELF, EXIT_FAILURE,
           "Unexpected error occured!", "");

  return 0;
}

int size(int isTmesh)
{
  if(nbid[1] > -1)
    return isTmesh ? nbid[1] : nbid[0];
  else
    return nbid[0];
}

bool useDerivedMeshBoundaryConditions()
{
  if (importFromNek) {
    return true;
  }
  else {
    return meshConditionsDerived;
  }
}


void check(mesh_t* mesh)
{
  
  int nid = nbid[0];
  if(mesh->cht) nid = nbid[1];

  int err = 0;
  int found = 0;

  for (int id = 1; id <= nid; id++) {
    found = 0;
    for (int f = 0; f < mesh->Nelements * mesh->Nfaces; f++) {
      if (mesh->EToB[f] == id) {
        found = 1;
        break;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &found, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm);
    err += (found ? 0 : 1);
    if (err && platform->comm.mpiRank == 0) 
      printf("Cannot find boundary ID %d in mesh!\n", id);
  }
  nrsCheck(err, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "");

  found = 0;
  for (int f = 0; f < mesh->Nelements * mesh->Nfaces; f++) {
    if (mesh->EToB[f] < -1 || mesh->EToB[f] == 0 || mesh->EToB[f] > nid)
      found = 1;
  }
  MPI_Allreduce(MPI_IN_PLACE, &found, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm);
  nrsCheck(found, platform->comm.mpiComm, EXIT_FAILURE, "%\n", "Mesh has unmapped boundary IDs!");
}

void setBcMap(std::string field, int* map, int nIDs)
{
  if (field.compare(0, 8, "scalar00") == 0)
    nbid[1] = nIDs;
  else
    nbid[0] = nIDs;

  fields.insert(field);
  for (int i = 0; i < nIDs; i++)
    bToBc[make_pair(field, i)] = map[i];
}

void checkBoundaryAlignment(mesh_t *mesh)
{
  int nid = nbid[0];
  if (mesh->cht)
    nid = nbid[1];

  bool bail = false;
  for (auto &&field : fields) {
    if (field != std::string("velocity") && field != std::string("mesh"))
      continue;

    std::map<int, boundaryAlignment_t> expectedAlignmentInvalidBIDs;
    std::map<int, std::set<boundaryAlignment_t>> actualAlignmentsInvalidBIDs;

    for (int e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        int bid = mesh->EToB[e * mesh->Nfaces + f];
        int bc = id(bid, field);
        if (bc == bcTypeSYMX || bc == bcTypeSYMY || bc == bcTypeSYMZ || bc == bcTypeSHLX ||
            bc == bcTypeSHLY || bc == bcTypeSHLZ || bc == bcTypeONX || bc == bcTypeONY || bc == bcTypeONZ) {
          auto expectedAlignment = boundaryAlignment_t::UNALIGNED;
          switch (bc) {
          case bcTypeSYMX:
            expectedAlignment = boundaryAlignment_t::X;
            break;
          case bcTypeSHLX:
            expectedAlignment = boundaryAlignment_t::X;
            break;
          case bcTypeONX:
            expectedAlignment = boundaryAlignment_t::X;
            break;
          case bcTypeSYMY:
            expectedAlignment = boundaryAlignment_t::Y;
            break;
          case bcTypeSHLY:
            expectedAlignment = boundaryAlignment_t::Y;
            break;
          case bcTypeONY:
            expectedAlignment = boundaryAlignment_t::Y;
            break;
          case bcTypeSYMZ:
            expectedAlignment = boundaryAlignment_t::Z;
            break;
          case bcTypeSHLZ:
            expectedAlignment = boundaryAlignment_t::Z;
            break;
          case bcTypeONZ:
            expectedAlignment = boundaryAlignment_t::Z;
            break;
          }

          auto alignment = computeAlignment(mesh, e, f);
          if (alignment != expectedAlignment) {
            expectedAlignmentInvalidBIDs[bid] = expectedAlignment;
            actualAlignmentsInvalidBIDs[bid].insert(alignment);
          }
        }
      }
    }

    int err = expectedAlignmentInvalidBIDs.size();
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm);
    if (err > 0) {
      bail = true;

      std::vector<int> valid(nid, 1);
      for (int bid = 1; bid <= nid; bid++) {
        valid[bid - 1] = expectedAlignmentInvalidBIDs.count(bid) == 0;
      }

      constexpr int invalidAlignment = -1;
      constexpr int nAlignments = 4;
      std::vector<int> expectedAlignments(nid, invalidAlignment);
      std::vector<int> encounteredAlignments(nid * nAlignments, invalidAlignment);
      for (auto &&bidAndAlignments : actualAlignmentsInvalidBIDs) {
        const auto bid = bidAndAlignments.first;
        const auto &alignments = bidAndAlignments.second;
        encounteredAlignments[(bid - 1) * nAlignments + 0] = (alignments.count(boundaryAlignment_t::X));
        encounteredAlignments[(bid - 1) * nAlignments + 1] = (alignments.count(boundaryAlignment_t::Y));
        encounteredAlignments[(bid - 1) * nAlignments + 2] = (alignments.count(boundaryAlignment_t::Z));
        encounteredAlignments[(bid - 1) * nAlignments + 3] =
            (alignments.count(boundaryAlignment_t::UNALIGNED));
        expectedAlignments[(bid - 1)] = static_cast<int>(expectedAlignmentInvalidBIDs[bid]);
      }
      MPI_Allreduce(MPI_IN_PLACE, valid.data(), nid, MPI_INT, MPI_MIN, platform->comm.mpiComm);
      MPI_Allreduce(MPI_IN_PLACE,
                    encounteredAlignments.data(),
                    nid * nAlignments,
                    MPI_INT,
                    MPI_MAX,
                    platform->comm.mpiComm);
      MPI_Allreduce(MPI_IN_PLACE, expectedAlignments.data(), nid, MPI_INT, MPI_MAX, platform->comm.mpiComm);

      if (platform->comm.mpiRank == 0) {
        std::cout << "Encountered incorrectly aligned boundaries in field \"" << field << "\":\n";
        for (int bid = 1; bid <= nid; bid++) {
          if (valid[bid - 1] == 0) {
            std::cout << "\tBoundary ID " << bid << ":\n";
            std::cout << "\t\texpected alignment : "
                      << to_string(static_cast<boundaryAlignment_t>(expectedAlignments[bid - 1])) << "\n";
            std::cout << "\t\tencountered alignments:\n";
            if (encounteredAlignments[(bid - 1) * nAlignments + 0])
              std::cout << "\t\t\tX\n";
            if (encounteredAlignments[(bid - 1) * nAlignments + 1])
              std::cout << "\t\t\tY\n";
            if (encounteredAlignments[(bid - 1) * nAlignments + 2])
              std::cout << "\t\t\tZ\n";
            if (encounteredAlignments[(bid - 1) * nAlignments + 3])
              std::cout << "\t\t\tUNALIGNED\n";
          }
        }
      }

      fflush(stdout);
      MPI_Barrier(platform->comm.mpiComm);
    }
  }

  nrsCheck(bail, platform->comm.mpiComm, EXIT_FAILURE, "\n", "");
}

void remapUnalignedBoundaries(mesh_t *mesh)
{
  return; // disable to avoid invalid combinations

  for (auto &&field : fields) {
    if (field != std::string("velocity") && field != std::string("mesh"))
      continue;

    std::map<int, bool> remapBID;
    std::map<int, boundaryAlignment_t> alignmentBID;

    int nid = nbid[0];
    if (mesh->cht)
      nid = nbid[1];

    for (int bid = 1; bid <= nid; ++bid) {
      int bcType = id(bid, field);
      remapBID[bid] = (bcType == bcTypeSYM);
    }

    for (int e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        int bid = mesh->EToB[f + e * mesh->Nfaces];
        int bc = id(bid, field);
        auto alignment = computeAlignment(mesh, e, f);
        if (alignmentBID.count(bid) == 0) {
          alignmentBID[bid] = alignment;
        }

        auto previousAlignment = alignmentBID[bid];
        remapBID[bid] &= (alignment != boundaryAlignment_t::UNALIGNED) && (alignment == previousAlignment);
      }
    }

    // if a single unaligned boundary with SYM/SHL is present, no remapping may occur.
    int unalignedBoundaryPresent = 0;
    for (int bid = 1; bid <= nid; ++bid) {
      int canRemap = remapBID[bid];
      int bc = id(bid, field);
      bool unalignedBoundaryType = bc == bcTypeSYM || bc == bcTypeSHL;
      if (!canRemap && unalignedBoundaryType) {
        unalignedBoundaryPresent++;
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &unalignedBoundaryPresent, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm);
    if (unalignedBoundaryPresent > 0) {
      return;
    }

    for (int bid = 1; bid <= nid; ++bid) {
      int canRemap = remapBID[bid];
      MPI_Allreduce(MPI_IN_PLACE, &canRemap, 1, MPI_INT, MPI_MIN, platform->comm.mpiComm);
      if (canRemap) {
        if(platform->comm.mpiRank == 0 && platform->options.compareArgs("VERBOSE","TRUE")){
          std::cout << "Remapping bid " << bid << " to an aligned type!\n";
        }

        auto alignmentType = alignmentBID[bid];

        int newBcType = 0;
        switch (alignmentType) {
        case boundaryAlignment_t::X:
          newBcType = bcTypeSYMX;
          break;
        case boundaryAlignment_t::Y:
          newBcType = bcTypeSYMY;
          break;
        case boundaryAlignment_t::Z:
          newBcType = bcTypeSYMZ;
          break;
        default:
          break;
        }

        bToBc.at({field, bid - 1}) = newBcType;
      }
    }
  }
}

bool unalignedMixedBoundary(std::string field)
{
  int nid = nbid[0];
  if (field.compare("mesh") == 0)
    nid = (nbid[1] > -1) ? nbid[1] : nbid[0]; 

  for (int bid = 1; bid <= nid; bid++) {
    int bcType = id(bid, field);
    if (bcType == bcTypeSYM)
      return true;
    if (bcType == bcTypeSHL)
      return true;
    if (bcType == bcTypeON) 
      return true;
  }

  return false;
}

} // namespace
