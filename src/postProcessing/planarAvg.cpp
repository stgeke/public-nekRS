#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "nekInterfaceAdapter.hpp"
#include "postProcessing.hpp"

namespace {
inline int mod1(int i, int n) {
  if(!i) return 0;
  return (i+n-1)%n + 1;
}

void get_exyz(int& ex, int& ey, int& ez,int eg, int nelx, int nely)
{
  ex = mod1(eg, nelx);
  ey = 1 + (mod1(eg, nelx*nely) - 1)/nelx;
  ez = 1 + (eg-1)/(nelx*nely);
}

oogs_t *gtpp_gs_setup(nrs_t *nrs, int nelgx, int nelgy, int nelgz, std::string dir)
{
  mesh_t* mesh = nrs->meshV;
  const auto nelgxy = nelgx*nelgy;
  const auto nelgyz = nelgy*nelgz;
  const auto nelgzx = nelgz*nelgx;

  auto *ids = (hlong *) calloc(mesh->Nlocal, sizeof(hlong)); 

  for(int iel = 0; iel < mesh->Nelements; iel++) {
    const auto eg = nek::lglel(iel) + 1;
    int ex, ey, ez;
    const auto nx1 = mesh->Nq;
    const auto ny1 = mesh->Nq;
    const auto nz1 = mesh->Nq;

    // Enumerate points in the y-z plane
    if(dir == "x") {
       get_exyz(ex,ey,ez,eg,nelgx,nelgyz);
       const auto ex_g = ey;
       for(int k = 0; k < mesh->Nq; k++) {
         for(int j = 0; j < mesh->Nq; j++) {
           for(int i = 0; i < mesh->Nq; i++) {
             const auto id = iel*mesh->Np + k*mesh->Nq*mesh->Nq + j*mesh->Nq + i; 
             ids[id] = (j+1) + ny1*k + ny1*nz1*(ex_g-1);
           }
         }
       } 
    }

    // Enumerate points in the x-z plane
    if(dir == "y") {
       get_exyz(ex,ey,ez,eg,nelgx,nelgy);
       const auto ex_g = (ez-1)*nelgx+ex;
       for(int k = 0; k < mesh->Nq; k++) {
         for(int j = 0; j < mesh->Nq; j++) {
           for(int i = 0; i < mesh->Nq; i++) {
             const auto id = iel*mesh->Np + k*mesh->Nq*mesh->Nq + j*mesh->Nq + i; 
             ids[id] = (k+1) + nz1*i + nx1*nz1*(ex_g-1);
           }
         }
       } 
    }

    // Enumerate points in the x-y plane
    if(dir == "z") {
       get_exyz(ex,ey,ez,eg,nelgxy,1);
       const auto ex_g = ex;
       for(int k = 0; k < mesh->Nq; k++) {
         for(int j = 0; j < mesh->Nq; j++) {
           for(int i = 0; i < mesh->Nq; i++) {
             const auto id = iel*mesh->Np + k*mesh->Nq*mesh->Nq + j*mesh->Nq + i; 
             ids[id] = (i+1) + nx1*j + nx1*ny1*(ex_g-1) + 1;
           }
         }
       } 
    }

    if(dir == "xz" || dir == "zx") {
       get_exyz(ex,ey,ez,eg,nelgx,nelgyz);
       for(int k = 0; k < mesh->Nq; k++) {
         for(int j = 0; j < mesh->Nq; j++) {
           for(int i = 0; i < mesh->Nq; i++) {
             const auto id = iel*mesh->Np + k*mesh->Nq*mesh->Nq + j*mesh->Nq + i; 
             ids[id] = (j+1) + nx1*(ey-1);
           }
         }
       } 
    }

    if(dir == "yz" || dir == "zy") {
       get_exyz(ex,ey,ez,eg,nelgx,nelgyz);
       for(int k = 0; k < mesh->Nq; k++) {
         for(int j = 0; j < mesh->Nq; j++) {
           for(int i = 0; i < mesh->Nq; i++) {
             const auto id = iel*mesh->Np + k*mesh->Nq*mesh->Nq + j*mesh->Nq + i; 
             ids[id] = (i+1) + nx1*(ex-1);
           }
         }
       } 
    }
    if(dir == "xy" || dir == "yx") {
       get_exyz(ex,ey,ez,eg,nelgx,nelgyz);
       for(int k = 0; k < mesh->Nq; k++) {
         for(int j = 0; j < mesh->Nq; j++) {
           for(int i = 0; i < mesh->Nq; i++) {
             const auto id = iel*mesh->Np + k*mesh->Nq*mesh->Nq + j*mesh->Nq + i; 
             ids[id] = (k+1) + nx1*(ez-1);
           }
         }
       } 
    }
  }

#if 0
  dfloat *idsDfloat = (dfloat *) calloc(mesh->Nlocal, sizeof(dfloat));
  for(int i = 0; i < mesh->Nlocal; i++) idsDfloat[i] = ids[i];
  occa::memory o_idsDfloat = platform->device.malloc(1*mesh->Nlocal * sizeof(dfloat), idsDfloat);
  writeFld("id" + dir, 0.0, 1, 1, &o_NULL, &o_NULL, &o_idsDfloat, 1);
#endif

  auto ogsh = ogsSetup(mesh->Nlocal, ids, platform->comm.mpiComm, 1, platform->device.occaDevice());
  free(ids);
  auto oogsh = oogs::setup(ogsh, 6, nrs->fieldOffset, ogsDfloat, NULL, OOGS_AUTO);
  return oogsh;
}

} // namespace

void postProcessing::planarAvg(nrs_t *nrs, const std::string& dir, int NELGX, int NELGY, int NELGZ, int nflds, occa::memory o_avg)
{
  mesh_t* mesh = nrs->meshV;
  const auto fieldOffsetByte = nrs->fieldOffset * sizeof(dfloat);

  static occa::memory o_avgWeight_x;
  static occa::memory o_avgWeight_y;
  static occa::memory o_avgWeight_z;

  static occa::memory o_avgWeight_xy;
  static occa::memory o_avgWeight_xz;
  static occa::memory o_avgWeight_yz;

  static oogs_t *oogs_x = nullptr;
  static oogs_t *oogs_y = nullptr;
  static oogs_t *oogs_z = nullptr;

  static oogs_t *oogs_xy = nullptr;
  static oogs_t *oogs_xz = nullptr;
  static oogs_t *oogs_yz = nullptr;

  occa::memory o_wghts;
  oogs_t *gsh;

  if(dir == "x") {
    o_wghts = o_avgWeight_x;
    gsh = oogs_x;
  } else if(dir == "y") {
    o_wghts = o_avgWeight_y;
    gsh = oogs_y;
  } else if(dir == "z") {
    o_wghts = o_avgWeight_z;
    gsh = oogs_z;
  } else if(dir == "xy" || dir == "yx") {
    o_wghts = o_avgWeight_xy;
    gsh = oogs_xy;
  } else if(dir == "xz" || dir == "zx") {
    o_wghts = o_avgWeight_xz;
    gsh = oogs_xz;
  } else if(dir == "yz" || dir == "zy") {
    o_wghts = o_avgWeight_yz;
    gsh = oogs_yz;
  } else {
    if (platform->comm.mpiRank == 0) printf("ERROR in planarAvg: Unknown direction!");
    ABORT(EXIT_FAILURE);
  }

  if(!gsh) {

    if(dir == "x"){
      oogs_x = gtpp_gs_setup(nrs, NELGX, NELGY, NELGZ, "x"); 
      gsh = oogs_x;
      o_avgWeight_x = platform->device.malloc(fieldOffsetByte);
      o_wghts = o_avgWeight_x;
    }
    else if(dir == "y"){
      oogs_y = gtpp_gs_setup(nrs, NELGX, NELGY, NELGZ, "y");
      gsh = oogs_y;
      o_avgWeight_y = platform->device.malloc(fieldOffsetByte);
      o_wghts = o_avgWeight_y;
    }
    else if(dir == "z"){
      oogs_z = gtpp_gs_setup(nrs, NELGX*NELGY, 1, NELGZ, "z");
      gsh = oogs_z;
      o_avgWeight_z = platform->device.malloc(fieldOffsetByte);
      o_wghts = o_avgWeight_z;
    }
    else if(dir == "xy" || dir == "yx"){
      oogs_xy = gtpp_gs_setup(nrs, NELGX, NELGY, NELGZ, "xy"); 
      gsh = oogs_xy;
      o_avgWeight_xy = platform->device.malloc(fieldOffsetByte);
      o_wghts = o_avgWeight_xy;
    }
    else if(dir == "xz" || dir == "zx"){
      oogs_xz = gtpp_gs_setup(nrs, NELGX, NELGY, NELGZ, "xz"); 
      gsh = oogs_xz;
      o_avgWeight_xz = platform->device.malloc(fieldOffsetByte);
      o_wghts = o_avgWeight_xz;
    }
    else if(dir == "yz" || dir == "zy"){
      oogs_yz = gtpp_gs_setup(nrs, NELGX, NELGY, NELGZ, "yz"); 
      gsh = oogs_yz;
      o_avgWeight_yz = platform->device.malloc(fieldOffsetByte);
      o_wghts = o_avgWeight_yz;
    }
    else{
      if (platform->comm.mpiRank == 0) printf("ERROR in planarAvg: Unknown direction!");
      ABORT(EXIT_FAILURE);
    }

    o_wghts.copyFrom(mesh->o_LMM, mesh->Nlocal*sizeof(dfloat));
    oogs::startFinish(o_wghts, 1, mesh->Nlocal, ogsDfloat, ogsAdd, gsh);
    platform->linAlg->ady(mesh->Nlocal, 1, o_wghts);
    platform->linAlg->axmy(mesh->Nlocal, 1, mesh->o_LMM, o_wghts);

  }

  for(int ifld = 0; ifld < nflds; ifld++) {
    auto o_wrk = o_avg.slice(ifld*fieldOffsetByte, fieldOffsetByte);
    platform->linAlg->axmy(mesh->Nlocal, 1, o_wghts, o_wrk);
  } 
  oogs::startFinish(o_avg, nflds, nrs->fieldOffset, ogsDfloat, ogsAdd, gsh);
} 
