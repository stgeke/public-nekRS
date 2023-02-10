/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

// for platform
#include "nrssys.hpp"
#include "nrs.hpp"

#include <cassert>
#include <cstdlib>
#include <vector>
#include "ogstypes.h"
#include "findpts.hpp"
#include "gslib.h"


// local data structures to switch between run-time/compile-time sizes
struct evalSrcPt_t {
  double r[findpts::dim];
  int index, proc, el;
};

template <int N> struct evalOutPt_t {
  double out[N];
  int index, proc;
};

// ======= Legacy setup =======
extern "C" {
struct hash_data_3 {
  ulong hash_n;
  struct dbl_range bnd[findpts::dim];
  double fac[findpts::dim];
  uint *offset;
};
struct findpts_data_3 {
  struct crystal cr;
  struct findpts_local_data_3 local;
  struct hash_data_3 hash;
};

auto *legacyFindptsSetup(MPI_Comm mpi_comm,
                         const dfloat *const elx[findpts::dim],
                         const dlong n[findpts::dim],
                         const dlong nel,
                         const dlong m[findpts::dim],
                         const dfloat bbox_tol,
                         const hlong local_hash_size,
                         const hlong global_hash_size,
                         const dlong npt_max,
                         const dfloat newt_tol)
{

  struct comm gs_comm;
  comm_init(&gs_comm, mpi_comm);

  const unsigned int n_[findpts::dim] = {(unsigned int)n[0], (unsigned int)n[1], (unsigned int)n[2]};
  const unsigned int m_[findpts::dim] = {(unsigned int)m[0], (unsigned int)m[1], (unsigned int)m[2]};
  auto *legacyFD = findpts_setup_3(&gs_comm,
                                   elx,
                                   n_,
                                   nel,
                                   m_,
                                   bbox_tol,
                                   local_hash_size,
                                   global_hash_size,
                                   npt_max,
                                   newt_tol);
  comm_free(&gs_comm);
  return legacyFD;
}
} // extern "C"
// ======= Legacy setup =======

namespace findpts {
namespace {

namespace pool {
static occa::memory o_scratch;
static occa::memory h_out;
static occa::memory h_r;
static occa::memory h_el;
static occa::memory h_dist2;
static occa::memory h_code;

static dfloat *out;
static dfloat *r;
static dlong *el;
static dfloat *dist2;
static dlong *code;

static void manageBuffers(dlong pn, dlong outputOffset, dlong nOutputFields)
{
  if (pn == 0)
    return;

  dlong Nbytes = 0;
  Nbytes += pn * sizeof(dlong);                  // code
  Nbytes += pn * sizeof(dlong);                  // element
  Nbytes += pn * sizeof(dfloat);                 // dist2
  Nbytes += dim * pn * sizeof(dfloat);           // r,s,t data
  Nbytes += dim * pn * sizeof(dfloat);           // x,y,z coordinates
  Nbytes += nOutputFields * outputOffset * sizeof(dfloat); // output buffer

  if (Nbytes > pool::o_scratch.size()) {
    if (pool::o_scratch.size())
      pool::o_scratch.free();
    void *buffer = std::calloc(Nbytes, 1);
    pool::o_scratch = platform->device.malloc(Nbytes, buffer);
    std::free(buffer);
  }

  occa::properties props;
  props["host"] = true;

  const auto NbytesR = dim * pn * sizeof(dfloat);
  if (NbytesR > pool::h_r.size()) {
    if (pool::h_r.size())
      pool::h_r.free();
    void *buffer = std::calloc(NbytesR, 1);
    pool::h_r = platform->device.malloc(NbytesR, buffer, props);
    pool::r = (dfloat *)pool::h_r.ptr();
    std::free(buffer);
  }

  const auto NbytesEl = pn * sizeof(dlong);
  if (NbytesEl > pool::h_el.size()) {
    if (pool::h_el.size())
      pool::h_el.free();
    void *buffer = std::calloc(NbytesEl, 1);
    pool::h_el = platform->device.malloc(NbytesEl, buffer, props);
    pool::el = (dlong *)pool::h_el.ptr();
    std::free(buffer);
  }

  const auto NbytesCode = pn * sizeof(dlong);
  if (NbytesCode > pool::h_code.size()) {
    if (pool::h_code.size())
      pool::h_code.free();
    void *buffer = std::calloc(NbytesCode, 1);
    pool::h_code = platform->device.malloc(NbytesCode, buffer, props);
    pool::code = (dlong *)pool::h_code.ptr();
    std::free(buffer);
  }

  const auto NbytesDist2 = pn * sizeof(dfloat);
  if (NbytesDist2 > pool::h_dist2.size()) {
    if (pool::h_dist2.size())
      pool::h_dist2.free();
    void *buffer = std::calloc(NbytesDist2, 1);
    pool::h_dist2 = platform->device.malloc(NbytesDist2, buffer, props);
    pool::dist2 = (dfloat *)pool::h_dist2.ptr();
    std::free(buffer);
  }

  const auto NbytesOut = nOutputFields * outputOffset * sizeof(dfloat);
  if (NbytesOut > pool::h_out.size() && NbytesOut > 0) {
    if (pool::h_out.size())
      pool::h_out.free();
    void *buffer = std::calloc(NbytesOut, 1);
    pool::h_out = platform->device.malloc(NbytesOut, buffer, props);
    pool::out = (dfloat *)pool::h_out.ptr();
    std::free(buffer);
  }
}
} // namespace pool
} // namespace

void findpts_t::findptsLocal(int *const code,
                             int *const el,
                             dfloat *const r,
                             dfloat *const dist2,
                             const dfloat *const x,
                             const dfloat *const y,
                             const dfloat *const z,
                             const int pn)
{
  if (pn == 0)
    return;

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.tic(timerName + "findpts_t::findptsLocal", 0);
  }

  pool::manageBuffers(pn, 0, 0);

  dlong byteOffset = 0;

  occa::memory o_code = pool::o_scratch + byteOffset;
  byteOffset += sizeof(dlong) * pn;

  occa::memory o_el = pool::o_scratch + byteOffset;
  byteOffset += sizeof(dlong) * pn;

  occa::memory o_r = pool::o_scratch + byteOffset;
  byteOffset += dim * sizeof(dfloat) * pn;

  occa::memory o_dist2 = pool::o_scratch + byteOffset;
  byteOffset += sizeof(dfloat) * pn;

  occa::memory o_xint = pool::o_scratch + byteOffset;
  byteOffset += sizeof(dfloat) * pn;

  occa::memory o_yint = pool::o_scratch + byteOffset;
  byteOffset += sizeof(dfloat) * pn;

  occa::memory o_zint = pool::o_scratch + byteOffset;
  byteOffset += sizeof(dfloat) * pn;

  o_xint.copyFrom(x, sizeof(dfloat) * pn);
  o_yint.copyFrom(y, sizeof(dfloat) * pn);
  o_zint.copyFrom(z, sizeof(dfloat) * pn);

  if (timerLevel != TimerLevel::None) {
    platform->timer.tic(timerName + "findpts_t::localKernel", 0);
  }
  this->localKernel(pn,
                    this->tol,
                    o_xint,
                    o_yint,
                    o_zint,
                    this->o_x,
                    this->o_y,
                    this->o_z,
                    this->o_wtend_x,
                    this->o_wtend_y,
                    this->o_wtend_z,
                    this->o_c,
                    this->o_A,
                    this->o_min,
                    this->o_max,
                    this->hash_n,
                    this->o_hashMin,
                    this->o_hashFac,
                    this->o_offset,
                    o_code,
                    o_el,
                    o_r,
                    o_dist2);
  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "findpts_t::localKernel");
  }

  o_code.copyTo(code, sizeof(dlong) * pn);
  o_el.copyTo(el, sizeof(dlong) * pn);
  o_r.copyTo(r, dim * sizeof(dfloat) * pn);
  o_dist2.copyTo(dist2, sizeof(dfloat) * pn);
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::findptsLocal");
  }
}

void findpts_t::findptsLocal(int *const code,
                             int *const el,
                             dfloat *const r,
                             dfloat *const dist2,
                             occa::memory o_xint,
                             occa::memory o_yint,
                             occa::memory o_zint,
                             const int pn)
{
  if (pn == 0)
    return;
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.tic(timerName + "findpts_t::findptsLocal", 0);
  }

  pool::manageBuffers(pn, 0, 0);

  dlong byteOffset = 0;

  occa::memory o_code = pool::o_scratch + byteOffset;
  byteOffset += sizeof(dlong) * pn;

  occa::memory o_el = pool::o_scratch + byteOffset;
  byteOffset += sizeof(dlong) * pn;

  occa::memory o_r = pool::o_scratch + byteOffset;
  byteOffset += dim * sizeof(dfloat) * pn;

  occa::memory o_dist2 = pool::o_scratch + byteOffset;
  byteOffset += sizeof(dfloat) * pn;

  if (timerLevel != TimerLevel::None) {
    platform->timer.tic(timerName + "findpts_t::localKernel", 0);
  }
  this->localKernel(pn,
                    this->tol,
                    o_xint,
                    o_yint,
                    o_zint,
                    this->o_x,
                    this->o_y,
                    this->o_z,
                    this->o_wtend_x,
                    this->o_wtend_y,
                    this->o_wtend_z,
                    this->o_c,
                    this->o_A,
                    this->o_min,
                    this->o_max,
                    this->hash_n,
                    this->o_hashMin,
                    this->o_hashFac,
                    this->o_offset,
                    o_code,
                    o_el,
                    o_r,
                    o_dist2);
  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "findpts_t::localKernel");
  }

  o_code.copyTo(code, sizeof(dlong) * pn);
  o_el.copyTo(el, sizeof(dlong) * pn);
  o_r.copyTo(r, dim * sizeof(dfloat) * pn);
  o_dist2.copyTo(dist2, sizeof(dfloat) * pn);
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::findptsLocal");
  }
}

template <typename OutputType>
void findpts_t::findptsLocalEvalInternal(OutputType *opt,
                                         const evalSrcPt_t *spt,
                                         const int pn,
                                         const int nFields,
                                         const int inputOffset,
                                         const int outputOffset,
                                         occa::memory &o_in)
{
  if (pn == 0)
    return;
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.tic(timerName + "findpts_t::findptsLocalEvalInternal", 0);
  }

  pool::manageBuffers(pn, outputOffset, nFields);

  dlong byteOffset = 0;

  auto o_out = pool::o_scratch;
  byteOffset += nFields * pn * sizeof(dfloat);

  auto o_r = pool::o_scratch + byteOffset;
  byteOffset += dim * pn * sizeof(dfloat);

  auto o_el = pool::o_scratch + byteOffset;
  byteOffset += pn * sizeof(dlong);

  // pack host buffers
  for (int point = 0; point < pn; ++point) {
    for (int component = 0; component < dim; ++component) {
      pool::r[dim * point + component] = spt[point].r[component];
    }
    pool::el[point] = spt[point].el;
  }

  o_r.copyFrom(pool::r, dim * pn * sizeof(dfloat));
  o_el.copyFrom(pool::el, pn * sizeof(dlong));

  if (timerLevel != TimerLevel::None) {
    platform->timer.tic(timerName + "findpts_t::localEvalKernel", 0);
  }
  this->localEvalKernel(pn, nFields, inputOffset, outputOffset, o_el, o_r, o_in, o_out);
  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "findpts_t::localEvalKernel");
  }

  o_out.copyTo(pool::out, nFields * outputOffset * sizeof(dfloat));

  // unpack buffer
  for (int point = 0; point < pn; ++point) {
    for (int field = 0; field < nFields; ++field) {
      opt[point].out[field] = pool::out[point + field * outputOffset];
    }
  }
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::findptsLocalEvalInternal");
  }
}

template <typename OutputType>
void findpts_t::findptsEvalImpl(occa::memory &o_out,
                                const int *const code_base,
                                const int *const proc_base,
                                const int *const el_base,
                                const dfloat *const r_base,
                                const int npt,
                                const int nFields,
                                const int inputOffset,
                                const int outputOffset,
                                occa::memory &o_in,
                                hashData_t &hash,
                                crystal &cr)
{
  int nGlobalRanks;
  MPI_Comm_size(platform->comm.mpiCommParent, &nGlobalRanks);

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.tic(timerName + "findpts_t::findptsEvalImpl", 1);
  }
  static std::vector<dfloat> out_base;

  if (out_base.size() < nFields * outputOffset) {
    constexpr int growthFactor = 2;
    out_base.resize(growthFactor * nFields * outputOffset);
  }

  struct array src, outpt;
  /* copy user data, weed out unfound points, send out */
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.tic(timerName + "findpts_t::findptsEvalImpl::copy data", 1);
  }
  {
    int index;
    const int *code = code_base, *proc = proc_base, *el = el_base;
    const dfloat *r = r_base;
    evalSrcPt_t *pt;
    array_init(evalSrcPt_t, &src, npt);
    pt = (evalSrcPt_t *)src.ptr;
    for (index = 0; index < npt; ++index) {
      if (*code != CODE_NOT_FOUND) {
        for (int d = 0; d < dim; ++d) {
          pt->r[d] = r[d];
        }
        pt->index = index;
        pt->proc = *proc;
        pt->el = *el;
        ++pt;
      }
      r += dim;
      code++;
      proc++;
      el++;
    }
    src.n = pt - (evalSrcPt_t *)src.ptr;
    if (nGlobalRanks > 1) {
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.tic(timerName + "findpts_t::findptsEvalImpl::copy data::sarray_transfer", 1);
      }
      sarray_transfer(evalSrcPt_t, &src, proc, 1, &cr);
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.toc(timerName + "findpts_t::findptsEvalImpl::copy data::sarray_transfer");
      }
    }
  }

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::findptsEvalImpl::copy data");
    platform->timer.tic(timerName + "findpts_t::findptsEvalImpl::eval points", 1);
  }

  /* evaluate points, send back */
  {
    int n = src.n;
    const evalSrcPt_t *spt;
    OutputType *opt;
    array_init(OutputType, &outpt, n);
    outpt.n = n;
    spt = (evalSrcPt_t *)src.ptr;
    opt = (OutputType *)outpt.ptr;
    findptsLocalEvalInternal(opt, spt, src.n, nFields, inputOffset, src.n, o_in);
    spt = (evalSrcPt_t *)src.ptr;
    opt = (OutputType *)outpt.ptr;
    for (; n; --n, ++spt, ++opt) {
      opt->index = spt->index;
      opt->proc = spt->proc;
    }
    array_free(&src);
    if (nGlobalRanks > 1) {
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.tic(timerName + "findpts_t::findptsEvalImpl::eval points::sarray_transfer", 1);
      }
      sarray_transfer(OutputType, &outpt, proc, 1, &cr);
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.toc(timerName + "findpts_t::findptsEvalImpl::eval points::sarray_transfer");
      }
    }
  }

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::findptsEvalImpl::eval points");
    platform->timer.tic(timerName + "findpts_t::findptsEvalImpl::copy results", 1);
  }

  /* copy results to user data */
  {
    int n = outpt.n;
    OutputType *opt = (OutputType *)outpt.ptr;
    for (; n; --n, ++opt) {
      for (int field = 0; field < nFields; ++field) {
        out_base[opt->index + outputOffset * field] = opt->out[field];
      }
    }
    o_out.copyFrom(out_base.data(), nFields * outputOffset * sizeof(dfloat));
    array_free(&outpt);
  }
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::findptsEvalImpl::copy results");
    platform->timer.toc(timerName + "findpts_t::findptsEvalImpl");
  }
}

template <typename OutputType>
void findpts_t::findptsEvalImpl(dfloat *out,
                                const int *const code_base,
                                const int *const proc_base,
                                const int *const el_base,
                                const dfloat *const r_base,
                                const int npt,
                                const int nFields,
                                const int inputOffset,
                                const int outputOffset,
                                const dfloat *const in,
                                hashData_t &hash,
                                crystal &cr)
{
  int nGlobalRanks;
  MPI_Comm_size(platform->comm.mpiCommParent, &nGlobalRanks);

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.tic(timerName + "findpts_t::findptsEvalImpl", 1);
  }

  static occa::memory o_in;

  const auto Nbytes = inputOffset * nFields * sizeof(dfloat);
  if (o_in.size() < Nbytes) {
    if (o_in.size())
      o_in.free();

    constexpr int growthFactor = 2;

    void *buffer = std::calloc(growthFactor * Nbytes, 1);
    o_in = platform->device.malloc(growthFactor * Nbytes, buffer);
    std::free(buffer);
  }

  o_in.copyFrom(in, Nbytes, 0, "async: true");

  struct array src, outpt;
  /* copy user data, weed out unfound points, send out */
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.tic(timerName + "findpts_t::findptsEvalImpl::copy data", 1);
  }
  {
    int index;
    const int *code = code_base, *proc = proc_base, *el = el_base;
    const dfloat *r = r_base;
    evalSrcPt_t *pt;
    array_init(evalSrcPt_t, &src, npt);
    pt = (evalSrcPt_t *)src.ptr;
    for (index = 0; index < npt; ++index) {
      if (*code != CODE_NOT_FOUND) {
        for (int d = 0; d < dim; ++d) {
          pt->r[d] = r[d];
        }
        pt->index = index;
        pt->proc = *proc;
        pt->el = *el;
        ++pt;
      }
      r += dim;
      code++;
      proc++;
      el++;
    }
    src.n = pt - (evalSrcPt_t *)src.ptr;
    if (nGlobalRanks > 1) {
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.tic(timerName + "findpts_t::findptsEvalImpl::copy data::sarray_transfer", 1);
      }
      sarray_transfer(evalSrcPt_t, &src, proc, 1, &cr);
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.toc(timerName + "findpts_t::findptsEvalImpl::copy data::sarray_transfer");
      }
    }
  }

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::findptsEvalImpl::copy data");
    platform->timer.tic(timerName + "findpts_t::findptsEvalImpl::eval points", 1);
  }

  /* evaluate points, send back */
  {
    int n = src.n;
    const evalSrcPt_t *spt;
    OutputType *opt;
    array_init(OutputType, &outpt, n);
    outpt.n = n;
    spt = (evalSrcPt_t *)src.ptr;
    opt = (OutputType *)outpt.ptr;

    // finish H->D transfer prior to findptsLocalEvalInternal call
    platform->device.finish();

    findptsLocalEvalInternal(opt, spt, src.n, nFields, inputOffset, src.n, o_in);
    spt = (evalSrcPt_t *)src.ptr;
    opt = (OutputType *)outpt.ptr;
    for (; n; --n, ++spt, ++opt) {
      opt->index = spt->index;
      opt->proc = spt->proc;
    }
    array_free(&src);
    if (nGlobalRanks > 1) {
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.tic(timerName + "findpts_t::findptsEvalImpl::eval points::sarray_transfer", 1);
      }
      sarray_transfer(OutputType, &outpt, proc, 1, &cr);
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.toc(timerName + "findpts_t::findptsEvalImpl::eval points::sarray_transfer");
      }
    }
  }

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::findptsEvalImpl::eval points");
    platform->timer.tic(timerName + "findpts_t::findptsEvalImpl::copy results", 1);
  }

  /* copy results to user data */
  {
    int n = outpt.n;
    OutputType *opt = (OutputType *)outpt.ptr;
    for (; n; --n, ++opt) {
      for (int field = 0; field < nFields; ++field) {
        out[opt->index + field * outputOffset] = opt->out[field];
      }
    }
    array_free(&outpt);
  }

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::findptsEvalImpl::copy results");
    platform->timer.toc(timerName + "findpts_t::findptsEvalImpl");
  }
}

extern "C" {
uint hash_opt_size_3(struct findpts_local_hash_data_3 *p,
                                   const struct obbox_3 *const obb,
                                   const uint nel,
                                   const uint max_size);
}

dlong getHashSize(const struct findpts_data_3 *fd, dlong nel, dlong max_hash_size)
{
  const findpts_local_data_3 *fd_local = &fd->local;
  auto hash_data_copy = fd_local->hd;
  return hash_opt_size_3(&hash_data_copy, fd_local->obb, nel, max_hash_size);
}

findpts_t::findpts_t(MPI_Comm comm,
                     const dfloat *const x,
                     const dfloat *const y,
                     const dfloat *const z,
                     const dlong Nq,
                     const dlong Nelements,
                     const dlong m,
                     const dfloat bbox_tol,
                     const hlong local_hash_size,
                     const hlong global_hash_size,
                     const dlong npt_max,
                     const dfloat newt_tol)
{
  static_assert(sizeof(dfloat) == sizeof(double), "findpts : dfloat must be double");
  static_assert(sizeof(dlong) == sizeof(int), "findpts : dlong must be int");

  const dlong Nlocal = Nq * Nq * Nq * Nelements;

  const dfloat *elx[dim] = {x, y, z};
  const int n[dim] = {Nq, Nq, Nq};
  const int ms[dim] = {m, m, m};

  this->_findptsData = legacyFindptsSetup(comm,
                                          elx,
                                          n,
                                          Nelements,
                                          ms,
                                          bbox_tol,
                                          local_hash_size,
                                          global_hash_size,
                                          npt_max,
                                          newt_tol);

  auto *findptsData = (findpts_data_3 *)this->_findptsData;

  this->comm = comm;
  MPI_Comm_rank(comm, &this->rank);

  this->tol = findptsData->local.tol;
  this->hash = &findptsData->hash;
  this->cr = &findptsData->cr;

  if (x != nullptr) {
    this->o_x = platform->device.malloc(Nlocal * sizeof(dfloat));
    this->o_y = platform->device.malloc(Nlocal * sizeof(dfloat));
    this->o_z = platform->device.malloc(Nlocal * sizeof(dfloat));

    this->o_x.copyFrom(x, Nlocal * sizeof(dfloat));
    this->o_y.copyFrom(y, Nlocal * sizeof(dfloat));
    this->o_z.copyFrom(z, Nlocal * sizeof(dfloat));
    std::vector<dfloat> c(dim * Nelements, 0.0);
    std::vector<dfloat> A(dim * dim * Nelements, 0.0);
    std::vector<dfloat> minBound(dim * Nelements, 0.0);
    std::vector<dfloat> maxBound(dim * Nelements, 0.0);

    for (int e = 0; e < Nelements; ++e) {
      auto box = findptsData->local.obb[e];

      c[dim * e + 0] = box.c0[0];
      c[dim * e + 1] = box.c0[1];
      c[dim * e + 2] = box.c0[2];

      minBound[dim * e + 0] = box.x[0].min;
      minBound[dim * e + 1] = box.x[1].min;
      minBound[dim * e + 2] = box.x[2].min;

      maxBound[dim * e + 0] = box.x[0].max;
      maxBound[dim * e + 1] = box.x[1].max;
      maxBound[dim * e + 2] = box.x[2].max;

      for (int i = 0; i < 9; ++i) {
        A[9 * e + i] = box.A[i];
      }
    }

    this->o_c = platform->device.malloc(c.size() * sizeof(dfloat));
    this->o_A = platform->device.malloc(A.size() * sizeof(dfloat));
    this->o_min = platform->device.malloc(minBound.size() * sizeof(dfloat));
    this->o_max = platform->device.malloc(maxBound.size() * sizeof(dfloat));

    this->o_c.copyFrom(c.data(), c.size() * sizeof(dfloat));
    this->o_A.copyFrom(A.data(), A.size() * sizeof(dfloat));
    this->o_min.copyFrom(minBound.data(), minBound.size() * sizeof(dfloat));
    this->o_max.copyFrom(maxBound.data(), maxBound.size() * sizeof(dfloat));
  }

  auto hash = findptsData->local.hd;
  dfloat hashMin[dim];
  dfloat hashFac[dim];
  for (int d = 0; d < dim; ++d) {
    hashMin[d] = hash.bnd[d].min;
    hashFac[d] = hash.fac[d];
  }
  this->hash_n = hash.hash_n;
  this->o_hashMin = platform->device.malloc(dim * sizeof(dfloat));
  this->o_hashFac = platform->device.malloc(dim * sizeof(dfloat));
  this->o_hashMin.copyFrom(hashMin, dim * sizeof(dfloat));
  this->o_hashFac.copyFrom(hashFac, dim * sizeof(dfloat));

  auto kernels = findpts_t::initFindptsKernels(Nq);
  this->localEvalKernel = kernels.at(0);
  this->localKernel = kernels.at(1);

  this->o_wtend_x = platform->device.malloc(6 * Nq * sizeof(dfloat));
  this->o_wtend_y = platform->device.malloc(6 * Nq * sizeof(dfloat));
  this->o_wtend_z = platform->device.malloc(6 * Nq * sizeof(dfloat));
  this->o_wtend_x.copyFrom(findptsData->local.fed.wtend[0], 6 * Nq * sizeof(dfloat));
  this->o_wtend_y.copyFrom(findptsData->local.fed.wtend[1], 6 * Nq * sizeof(dfloat));
  this->o_wtend_z.copyFrom(findptsData->local.fed.wtend[2], 6 * Nq * sizeof(dfloat));

  const auto hd_d_size = getHashSize(findptsData, Nelements, local_hash_size);

  std::vector<dlong> offsets(hd_d_size, 0);
  for (dlong i = 0; i < hd_d_size; ++i) {
    offsets[i] = findptsData->local.hd.offset[i];
  }
  this->o_offset = platform->device.malloc(offsets.size() * sizeof(dlong));
  this->o_offset.copyFrom(offsets.data(), offsets.size() * sizeof(dlong));
}

findpts_t::~findpts_t()
{
  auto *findptsData = (findpts_data_3 *)this->_findptsData;
  findpts_free_3(findptsData);
}

static slong lfloor(dfloat x) { return floor(x); }
static slong lceil(dfloat x) { return ceil(x); }

static ulong hash_index_aux(dfloat low, dfloat fac, ulong n, dfloat x)
{
  const slong i = lfloor((x - low) * fac);
  return i < 0 ? 0 : (n - 1 < (ulong)i ? n - 1 : (ulong)i);
}

static ulong hash_index_3(const hashData_t *p, const dfloat x[dim])
{
  const ulong n = p->hash_n;
  return (hash_index_aux(p->bnd[2].min, p->fac[2], n, x[2]) * n +
          hash_index_aux(p->bnd[1].min, p->fac[1], n, x[1])) *
             n +
         hash_index_aux(p->bnd[0].min, p->fac[0], n, x[0]);
}

struct srcPt_t {
  dfloat x[dim];
  int index, proc;
};
struct outPt_t {
  dfloat r[dim], dist2;
  int index, code, el, proc;
};

void findpts_t::find(data_t *const findPtsData,
                     occa::memory o_xint,
                     occa::memory o_yint,
                     occa::memory o_zint,
                     const dlong npt)
{
  int nGlobalRanks;
  MPI_Comm_size(platform->comm.mpiCommParent, &nGlobalRanks);

  if (timerLevel != TimerLevel::None) {
    platform->timer.tic(timerName + "findpts_t::find", 1);
  }

  static std::vector<dfloat> x_base;
  static std::vector<dfloat> y_base;
  static std::vector<dfloat> z_base;
  static std::vector<int> codeArr;
  static std::vector<int> elArr;
  static std::vector<dfloat> rArr;
  static std::vector<dfloat> dist2Arr;
  static std::vector<dfloat> x0;
  static std::vector<dfloat> x1;
  static std::vector<dfloat> x2;

  if (x_base.size() < npt) {
    constexpr int growthFactor = 2;
    x_base.resize(growthFactor * npt);
    y_base.resize(growthFactor * npt);
    z_base.resize(growthFactor * npt);
  }

  // start async D->H transfer of o_xint -> x_base, etc.
  o_xint.copyTo(x_base.data(), npt * sizeof(dfloat), 0, "async: true");
  o_yint.copyTo(y_base.data(), npt * sizeof(dfloat), 0, "async: true");
  o_zint.copyTo(z_base.data(), npt * sizeof(dfloat), 0, "async: true");

  int *const code_base = findPtsData->code_base;
  int *const proc_base = findPtsData->proc_base;
  int *const el_base = findPtsData->el_base;
  dfloat *const r_base = findPtsData->r_base;
  dfloat *const dist2_base = findPtsData->dist2_base;
  hashData_t &hash = *this->hash;
  crystal &cr = *this->cr;
  const int np = cr.comm.np, id = cr.comm.id;
  struct array hash_pt, srcPt_t, outPt_t;
  /* look locally first */
  if (npt) {
    findptsLocal(code_base, el_base, r_base, dist2_base, o_xint, o_yint, o_zint, npt);
  }

  // finish D->H transfer
  platform->device.finish();

  /* send unfound and border points to global hash cells */
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.tic(timerName + "findpts_t::find::unfound", 1);
  }
  {
    int index;
    int *code = code_base, *proc = proc_base;
    const dfloat *xp[dim];
    struct srcPt_t *pt;

    xp[0] = x_base.data();
    xp[1] = y_base.data();
    xp[2] = z_base.data();

    array_init(struct srcPt_t, &hash_pt, npt);
    pt = (struct srcPt_t *)hash_pt.ptr;

    dfloat x[dim];

    for (index = 0; index < npt; ++index) {
      for (int d = 0; d < dim; ++d) {
        x[d] = *xp[d];
      }
      *proc = id;
      if (*code != CODE_INTERNAL) {
        const int hi = hash_index_3(&hash, x);
        for (int d = 0; d < dim; ++d) {
          pt->x[d] = x[d];
        }
        pt->index = index;
        pt->proc = hi % np;
        ++pt;
      }
      for (int d = 0; d < dim; ++d) {
        xp[d]++;
      }
      code++;
      proc++;
    }
    hash_pt.n = pt - (struct srcPt_t *)hash_pt.ptr;
    if (nGlobalRanks > 1) {
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.tic(timerName + "findpts_t::find::unfound::sarray_transfer", 1);
      }
      sarray_transfer(struct srcPt_t, &hash_pt, proc, 1, &cr);
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.toc(timerName + "findpts_t::find::unfound::sarray_transfer");
      }
    }
  }

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::find::unfound");
    platform->timer.tic(timerName + "findpts_t::find::send unfound", 1);
  }

  /* look up points in hash cells, route to possible procs */
  {
    const unsigned int *const hash_offset = hash.offset;
    int count = 0, *proc, *proc_p;
    const struct srcPt_t *p = (struct srcPt_t *)hash_pt.ptr, *const pe = p + hash_pt.n;
    struct srcPt_t *q;
    for (; p != pe; ++p) {
      const int hi = hash_index_3(&hash, p->x) / np;
      const int i = hash_offset[hi], ie = hash_offset[hi + 1];
      count += ie - i;
    }
    proc = tmalloc(int, count);
    proc_p = proc;
    array_init(struct srcPt_t, &srcPt_t, count), q = (struct srcPt_t *)srcPt_t.ptr;
    p = (struct srcPt_t *)hash_pt.ptr;
    for (; p != pe; ++p) {
      const int hi = hash_index_3(&hash, p->x) / np;
      int i = hash_offset[hi];
      const int ie = hash_offset[hi + 1];
      for (; i != ie; ++i) {
        const int pp = hash_offset[i];
        if (pp == p->proc)
          continue; /* don't send back to source proc */
        *proc_p++ = pp;
        *q++ = *p;
      }
    }
    array_free(&hash_pt);
    srcPt_t.n = proc_p - proc;
#ifdef DIAGNOSTICS
    printf("(proc %u) hashed; routing %u/%u\n", id, (int)srcPt_t.n, count);
#endif
    if (nGlobalRanks > 1) {
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.tic(timerName + "findpts_t::find::send unfound::sarray_transfer_ext", 1);
      }
      sarray_transfer_ext(struct srcPt_t, &srcPt_t, reinterpret_cast<unsigned int *>(proc), sizeof(int), &cr);
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.toc(timerName + "findpts_t::find::send unfound::sarray_transfer_ext");
      }
    }
    free(proc);
  }

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::find::send unfound");
    platform->timer.tic(timerName + "findpts_t::find::send back", 1);
  }

  /* look for other procs' points, send back */
  {
    int n = srcPt_t.n;
    const struct srcPt_t *spt;
    struct outPt_t *opt;
    array_init(struct outPt_t, &outPt_t, n);
    outPt_t.n = n;
    spt = (struct srcPt_t *)srcPt_t.ptr;
    opt = (struct outPt_t *)outPt_t.ptr;
    for (; n; --n, ++spt, ++opt) {
      opt->index = spt->index;
      opt->proc = spt->proc;
    }
    spt = (struct srcPt_t *)srcPt_t.ptr;
    opt = (struct outPt_t *)outPt_t.ptr;
    if (srcPt_t.n) {

      // resize result buffers
      if (codeArr.size() < srcPt_t.n) {
        constexpr int growthFactor = 2;
        codeArr.resize(growthFactor * srcPt_t.n);
        elArr.resize(growthFactor * srcPt_t.n);
        rArr.resize(growthFactor * dim * srcPt_t.n);
        dist2Arr.resize(growthFactor * srcPt_t.n);

        x0.resize(growthFactor * srcPt_t.n);
        x1.resize(growthFactor * srcPt_t.n);
        x2.resize(growthFactor * srcPt_t.n);
      }

      for (int point = 0; point < srcPt_t.n; ++point) {
        x0[point] = spt[point].x[0];
        x1[point] = spt[point].x[1];
        x2[point] = spt[point].x[2];
      }

      findptsLocal(codeArr.data(),
                   elArr.data(),
                   rArr.data(),
                   dist2Arr.data(),
                   x0.data(),
                   x1.data(),
                   x2.data(),
                   srcPt_t.n);

      // unpack arrays into opt
      for (int point = 0; point < srcPt_t.n; point++) {
        opt[point].code = codeArr[point];
        opt[point].el = elArr[point];
        opt[point].dist2 = dist2Arr[point];
        for (int d = 0; d < dim; ++d) {
          opt[point].r[d] = rArr[dim * point + d];
        }
      }
    }
    array_free(&srcPt_t);
    /* group by code to eliminate unfound points */
    sarray_sort(struct outPt_t, opt, outPt_t.n, code, 0, &cr.data);
    n = outPt_t.n;
    while (n && opt[n - 1].code == CODE_NOT_FOUND)
      --n;
    outPt_t.n = n;
#ifdef DIAGNOSTICS
    printf("(proc %u) sending back %u found points\n", id, (int)outPt_t.n);
#endif
    if (nGlobalRanks > 1) {
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.tic(timerName + "findpts_t::find::send back::sarray_transfer", 1);
      }
      sarray_transfer(struct outPt_t, &outPt_t, proc, 1, &cr);
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.toc(timerName + "findpts_t::find::send back::sarray_transfer");
      }
    }
  }

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::find::send back");
    platform->timer.tic(timerName + "findpts_t::find::merge", 1);
  }

  /* merge remote results with user data */
  {
    int n = outPt_t.n;
    struct outPt_t *opt = (struct outPt_t *)outPt_t.ptr;
    for (; n; --n, ++opt) {
      const int index = opt->index;
      if (code_base[index] == CODE_INTERNAL)
        continue;
      if (code_base[index] == CODE_NOT_FOUND || opt->code == CODE_INTERNAL ||
          opt->dist2 < dist2_base[index]) {
        for (int d = 0; d < dim; ++d) {
          r_base[dim * index + d] = opt->r[d];
        }
        dist2_base[index] = opt->dist2;
        proc_base[index] = opt->proc;
        el_base[index] = opt->el;
        code_base[index] = opt->code;
      }
    }
    array_free(&outPt_t);
  }
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::find::merge");
  }
  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "findpts_t::find");
  }
}

void findpts_t::find(data_t *const findPtsData,
                     const dfloat *const x_base,
                     const dfloat *const y_base,
                     const dfloat *const z_base,
                     const dlong npt)
{
  int nGlobalRanks;
  MPI_Comm_size(platform->comm.mpiCommParent, &nGlobalRanks);
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic("findpts_t::find", 1);
  }
  static std::vector<int> codeArr;
  static std::vector<int> elArr;
  static std::vector<dfloat> rArr;
  static std::vector<dfloat> dist2Arr;
  static std::vector<dfloat> x0;
  static std::vector<dfloat> x1;
  static std::vector<dfloat> x2;

  int *const code_base = findPtsData->code_base;
  int *const proc_base = findPtsData->proc_base;
  int *const el_base = findPtsData->el_base;
  dfloat *const r_base = findPtsData->r_base;
  dfloat *const dist2_base = findPtsData->dist2_base;
  hashData_t &hash = *this->hash;
  crystal &cr = *this->cr;
  const int np = cr.comm.np, id = cr.comm.id;
  struct array hash_pt, srcPt_t, outPt_t;
  /* look locally first */
  if (npt) {
    findptsLocal(code_base, el_base, r_base, dist2_base, x_base, y_base, z_base, npt);
  }

  /* send unfound and border points to global hash cells */
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.tic(timerName + "findpts_t::find::send unfound", 1);
  }
  {
    int index;
    int *code = code_base, *proc = proc_base;
    const dfloat *xp[dim];
    struct srcPt_t *pt;

    xp[0] = x_base;
    xp[1] = y_base;
    xp[2] = z_base;

    array_init(struct srcPt_t, &hash_pt, npt);
    pt = (struct srcPt_t *)hash_pt.ptr;

    dfloat x[dim];

    for (index = 0; index < npt; ++index) {
      for (int d = 0; d < dim; ++d) {
        x[d] = *xp[d];
      }
      *proc = id;
      if (*code != CODE_INTERNAL) {
        const int hi = hash_index_3(&hash, x);
        for (int d = 0; d < dim; ++d) {
          pt->x[d] = x[d];
        }
        pt->index = index;
        pt->proc = hi % np;
        ++pt;
      }
      for (int d = 0; d < dim; ++d) {
        xp[d]++;
      }
      code++;
      proc++;
    }
    hash_pt.n = pt - (struct srcPt_t *)hash_pt.ptr;
    if (nGlobalRanks > 1) {
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.tic(timerName + "findpts_t::find::unfound::sarray_transfer", 1);
      }
      sarray_transfer(struct srcPt_t, &hash_pt, proc, 1, &cr);
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.toc(timerName + "findpts_t::find::unfound::sarray_transfer");
      }
    }
  }

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::find::send unfound");
    platform->timer.tic("findpts_t::find::send unfound", 1);
  }

  /* look up points in hash cells, route to possible procs */
  {
    const unsigned int *const hash_offset = hash.offset;
    int count = 0, *proc, *proc_p;
    const struct srcPt_t *p = (struct srcPt_t *)hash_pt.ptr, *const pe = p + hash_pt.n;
    struct srcPt_t *q;
    for (; p != pe; ++p) {
      const int hi = hash_index_3(&hash, p->x) / np;
      const int i = hash_offset[hi], ie = hash_offset[hi + 1];
      count += ie - i;
    }
    proc = tmalloc(int, count);
    proc_p = proc;
    array_init(struct srcPt_t, &srcPt_t, count), q = (struct srcPt_t *)srcPt_t.ptr;
    p = (struct srcPt_t *)hash_pt.ptr;
    for (; p != pe; ++p) {
      const int hi = hash_index_3(&hash, p->x) / np;
      int i = hash_offset[hi];
      const int ie = hash_offset[hi + 1];
      for (; i != ie; ++i) {
        const int pp = hash_offset[i];
        if (pp == p->proc)
          continue; /* don't send back to source proc */
        *proc_p++ = pp;
        *q++ = *p;
      }
    }
    array_free(&hash_pt);
    srcPt_t.n = proc_p - proc;
#ifdef DIAGNOSTICS
    printf("(proc %u) hashed; routing %u/%u\n", id, (int)srcPt_t.n, count);
#endif
    if (nGlobalRanks > 1) {
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.tic(timerName + "findpts_t::find::send unfound::sarray_transfer_ext", 1);
      }
      sarray_transfer_ext(struct srcPt_t, &srcPt_t, reinterpret_cast<unsigned int *>(proc), sizeof(int), &cr);
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.toc(timerName + "findpts_t::find::send unfound::sarray_transfer_ext");
      }
    }
    free(proc);
  }

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::find::send unfound");
    platform->timer.tic(timerName + "findpts_t::find::send back", 1);
  }

  /* look for other procs' points, send back */
  {
    int n = srcPt_t.n;
    const struct srcPt_t *spt;
    struct outPt_t *opt;
    array_init(struct outPt_t, &outPt_t, n);
    outPt_t.n = n;
    spt = (struct srcPt_t *)srcPt_t.ptr;
    opt = (struct outPt_t *)outPt_t.ptr;
    for (; n; --n, ++spt, ++opt) {
      opt->index = spt->index;
      opt->proc = spt->proc;
    }
    spt = (struct srcPt_t *)srcPt_t.ptr;
    opt = (struct outPt_t *)outPt_t.ptr;
    if (srcPt_t.n) {

      // resize result buffers
      if (codeArr.size() < srcPt_t.n) {
        constexpr int growthFactor = 2;
        codeArr.resize(growthFactor * srcPt_t.n);
        elArr.resize(growthFactor * srcPt_t.n);
        rArr.resize(growthFactor * dim * srcPt_t.n);
        dist2Arr.resize(growthFactor * srcPt_t.n);

        x0.resize(growthFactor * srcPt_t.n);
        x1.resize(growthFactor * srcPt_t.n);
        x2.resize(growthFactor * srcPt_t.n);
      }

      for (int point = 0; point < srcPt_t.n; ++point) {
        x0[point] = spt[point].x[0];
        x1[point] = spt[point].x[1];
        x2[point] = spt[point].x[2];
      }

      findptsLocal(codeArr.data(),
                   elArr.data(),
                   rArr.data(),
                   dist2Arr.data(),
                   x0.data(),
                   x1.data(),
                   x2.data(),
                   srcPt_t.n);

      // unpack arrays into opt
      for (int point = 0; point < srcPt_t.n; point++) {
        opt[point].code = codeArr[point];
        opt[point].el = elArr[point];
        opt[point].dist2 = dist2Arr[point];
        for (int d = 0; d < dim; ++d) {
          opt[point].r[d] = rArr[dim * point + d];
        }
      }
    }
    array_free(&srcPt_t);
    /* group by code to eliminate unfound points */
    sarray_sort(struct outPt_t, opt, outPt_t.n, code, 0, &cr.data);
    n = outPt_t.n;
    while (n && opt[n - 1].code == CODE_NOT_FOUND)
      --n;
    outPt_t.n = n;
#ifdef DIAGNOSTICS
    printf("(proc %u) sending back %u found points\n", id, (int)outPt_t.n);
#endif
    if (nGlobalRanks > 1) {
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.tic(timerName + "findpts_t::find::send back::sarray_transfer", 1);
      }
      sarray_transfer(struct outPt_t, &outPt_t, proc, 1, &cr);
      if (timerLevel == TimerLevel::Detailed) {
        platform->timer.toc(timerName + "findpts_t::find::send back::sarray_transfer");
      }
    }
  }

  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::find::send back");
    platform->timer.tic(timerName + "findpts_t::find::merge", 1);
  }

  /* merge remote results with user data */
  {
    int n = outPt_t.n;
    struct outPt_t *opt = (struct outPt_t *)outPt_t.ptr;
    for (; n; --n, ++opt) {
      const int index = opt->index;
      if (code_base[index] == CODE_INTERNAL)
        continue;
      if (code_base[index] == CODE_NOT_FOUND || opt->code == CODE_INTERNAL ||
          opt->dist2 < dist2_base[index]) {
        for (int d = 0; d < dim; ++d) {
          r_base[dim * index + d] = opt->r[d];
        }
        dist2_base[index] = opt->dist2;
        proc_base[index] = opt->proc;
        el_base[index] = opt->el;
        code_base[index] = opt->code;
      }
    }
    array_free(&outPt_t);
  }
  if (timerLevel == TimerLevel::Detailed) {
    platform->timer.toc(timerName + "findpts_t::find::merge");
  }
  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "findpts_t::find");
  }
}

void findpts_t::eval(const dlong npt, occa::memory o_in, data_t *findPtsData, occa::memory o_out)
{
  this->eval(npt, 1, 0, npt, o_in, findPtsData, o_out);
}

void findpts_t::eval(const dlong npt, dfloat *in, data_t *findPtsData, dfloat *out)
{
  this->eval(npt, 1, 0, npt, in, findPtsData, out);
}

void findpts_t::eval(const dlong npt,
                     const dlong nFields,
                     const dlong inputOffset,
                     const dlong outputOffset,
                     occa::memory o_in,
                     data_t *findPtsData,
                     occa::memory o_out)
{
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic(timerName + "findpts_t::eval", 1);
  }
#define FINDPTS_EVAL(fieldSize)                                                                              \
{                                                                                                            \
if (nFields == (fieldSize)) {                                                                                \
findptsEvalImpl<evalOutPt_t<(fieldSize)>>(o_out,                                                             \
                                          findPtsData->code_base,                                            \
                                          findPtsData->proc_base,                                            \
                                          findPtsData->el_base,                                              \
                                          findPtsData->r_base,                                               \
                                          npt,                                                               \
                                          nFields,                                                           \
                                          inputOffset,                                                       \
                                          outputOffset,                                                      \
                                          o_in,                                                              \
                                          *this->hash,                                                       \
                                          *this->cr);                                                        \
}                                                                                                            \
}
  FINDPTS_EVAL(1);
  FINDPTS_EVAL(2);
  FINDPTS_EVAL(3);
  FINDPTS_EVAL(4);
  FINDPTS_EVAL(5);
  FINDPTS_EVAL(6);
  FINDPTS_EVAL(7);
  FINDPTS_EVAL(8);
  FINDPTS_EVAL(9);
  FINDPTS_EVAL(10);
#undef FINDPTS_EVAL

  if (nFields < 1 || nFields > 10) {
    if (this->rank == 0) {
      printf("Error: nFields = %d is not supported.\n", nFields);
    }
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "findpts_t::eval");
  }
}

void findpts_t::eval(const dlong npt,
                     const dlong nFields,
                     const dlong inputOffset,
                     const dlong outputOffset,
                     dfloat *in,
                     data_t *findPtsData,
                     dfloat *out)
{
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic(timerName + "findpts_t::eval", 1);
  }
#define FINDPTS_EVAL(fieldSize)                                                                              \
{                                                                                                            \
if (nFields == (fieldSize)) {                                                                                \
findptsEvalImpl<evalOutPt_t<(fieldSize)>>(out,                                                               \
                                          findPtsData->code_base,                                            \
                                          findPtsData->proc_base,                                            \
                                          findPtsData->el_base,                                              \
                                          findPtsData->r_base,                                               \
                                          npt,                                                               \
                                          nFields,                                                           \
                                          inputOffset,                                                       \
                                          outputOffset,                                                      \
                                          in,                                                                \
                                          *this->hash,                                                       \
                                          *this->cr);                                                        \
}                                                                                                            \
}
  FINDPTS_EVAL(1);
  FINDPTS_EVAL(2);
  FINDPTS_EVAL(3);
  FINDPTS_EVAL(4);
  FINDPTS_EVAL(5);
  FINDPTS_EVAL(6);
  FINDPTS_EVAL(7);
  FINDPTS_EVAL(8);
  FINDPTS_EVAL(9);
  FINDPTS_EVAL(10);
#undef FINDPTS_EVAL

  if (nFields < 1 || nFields > 10) {
    if (this->rank == 0) {
      printf("Error: nFields = %d is not supported.\n", nFields);
    }
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (timerLevel != TimerLevel::None) {
    platform->timer.toc(timerName + "findpts_t::eval");
  }
}

crystal *findpts_t::crystalRouter() { return this->cr; }

} // namespace findpts
