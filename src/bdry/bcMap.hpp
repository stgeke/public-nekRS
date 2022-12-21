#if !defined(nekrs_bcmap_hpp_)
#define nekrs_bcmap_hpp_

#include <string>
#include <vector>
#include "nekInterfaceAdapter.hpp"


namespace bcMap
{

// lower id wins
constexpr int bcTypeW = 1;
constexpr int bcTypeV = 2;
constexpr int bcTypeSYMX = 3;
constexpr int bcTypeSYMY = 4;
constexpr int bcTypeSYMZ = 5;
constexpr int bcTypeSYM = 6;
constexpr int bcTypeSHL = 7;
constexpr int bcTypeO = 8;

constexpr int bcTypeS = 1;
constexpr int bcTypeF0 = 2;
constexpr int bcTypeF = 3;

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
bool unalignedBoundary(bool cht, std::string field);
void deriveMeshBoundaryConditions(std::vector<std::string> velocityBCs);
bool useDerivedMeshBoundaryConditions();
}

#endif
