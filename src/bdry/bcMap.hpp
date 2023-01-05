#if !defined(nekrs_bcmap_hpp_)
#define nekrs_bcmap_hpp_

#include <string>
#include <vector>
#include "nekInterfaceAdapter.hpp"
#include "bcType.h"

namespace bcMap
{

constexpr int bcTypeW = p_bcTypeW;
constexpr int bcTypeV = p_bcTypeV;

constexpr int bcTypeSYMX = p_bcTypeSYMX;
constexpr int bcTypeSYMY = p_bcTypeSYMY;
constexpr int bcTypeSYMZ = p_bcTypeSYMZ;
constexpr int bcTypeSYM  = p_bcTypeSYM;

constexpr int bcTypeSHLX = p_bcTypeSHLX;
constexpr int bcTypeSHLY = p_bcTypeSHLY;
constexpr int bcTypeSHLZ = p_bcTypeSHLZ;
constexpr int bcTypeSHL  = p_bcTypeSHL;

constexpr int bcTypeONX = p_bcTypeONX;
constexpr int bcTypeONY = p_bcTypeONY;
constexpr int bcTypeONZ = p_bcTypeONZ;

constexpr int bcTypeON = p_bcTypeON;
constexpr int bcTypeO = p_bcTypeO;

constexpr int bcTypeS = p_bcTypeS;
constexpr int bcTypeF0 = p_bcTypeF0;
constexpr int bcTypeF = p_bcTypeF;

bool useNekBCs();
void setup(std::vector<std::string> slist, std::string field);
int id(int bid, std::string field);
int ellipticType(int bid, std::string field);
std::string text(int bid, std::string field);
int size(int isTmesh);
void check(mesh_t* mesh);
void setBcMap(std::string field, int* map, int nbid);
void checkBoundaryAlignment(mesh_t *mesh);
void remapUnalignedBoundaries(mesh_t *mesh);
bool unalignedRobinBoundary(std::string field);
void deriveMeshBoundaryConditions(std::vector<std::string> velocityBCs);
bool useDerivedMeshBoundaryConditions();
}

#endif
