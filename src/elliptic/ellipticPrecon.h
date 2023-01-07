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

#include "nrssys.hpp"
#include "MGSolver/MGSolver.hpp"
#include "SEMFEMSolver.hpp"
#include "parseMultigridSchedule.hpp"

struct nonZero_t
{
  hlong row;
  hlong col;
  int ownerRank;
  dfloat val;
};

class precon_t;

struct precon_t
{
  long long int preconBytes;

  occa::kernel coarsenKernel;
  occa::kernel prolongateKernel;

  occa::memory o_diagA;
  occa::memory o_invDiagA;

  MGSolver_t* MGSolver = nullptr;
  SEMFEMSolver_t* SEMFEMSolver = nullptr;

  ~precon_t();

  bool additive;
};

static std::vector<int> determineMGLevels(std::string section)
{
  const std::string optionsPrefix = [section]() {
    std::string prefix = section + std::string(" ");
    if (section.find("temperature") != std::string::npos) {
      prefix = std::string("scalar00 ");
    }
    std::transform(prefix.begin(), prefix.end(), prefix.begin(), [](unsigned char c) {
      return std::toupper(c);
    });
    return prefix;
  }();

  std::vector<int> levels;
  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);

  std::string p_mgschedule = platform->options.getArgs(optionsPrefix + "MULTIGRID SCHEDULE");
  if(!p_mgschedule.empty()){

    // note: default order is not required here.
    // We just need the levels, not the degree.
    auto [scheduleMap, errorString] = parseMultigridSchedule(p_mgschedule, platform->options, 3);
    for(auto && [cyclePosition, smootherOrder] : scheduleMap){
      auto [order, isDownLeg] = cyclePosition;
      if(isDownLeg){
        levels.push_back(order);
      }
    }

    std::sort(levels.rbegin(), levels.rend());

    if (levels.back() > 1) {
      if (platform->options.compareArgs(optionsPrefix + "MULTIGRID COARSE SOLVE", "TRUE")) {
        // if the coarse level has p > 1 and requires solving the coarsest level,
        // rather than just smoothing, SEMFEM must be used for the discretization
        const auto usesSEMFEM =
            platform->options.compareArgs(optionsPrefix + "MULTIGRID SEMFEM", "TRUE");

        if (!usesSEMFEM) {
          if (platform->comm.mpiRank == 0) {
            printf("Error! FEM coarse discretization only supports p=1 for the coarsest level!\n");
          }
          ABORT(1);
        }
      }
    }

    return levels;
  }

  if (platform->options.compareArgs(optionsPrefix + "MULTIGRID SMOOTHER", "ASM") ||
           platform->options.compareArgs(optionsPrefix + "MULTIGRID SMOOTHER", "RAS")) {
    std::map<int, std::vector<int>> mg_level_lookup = {
        {1, {1}},
        {2, {2, 1}},
        {3, {3, 1}},
        {4, {4, 2, 1}},
        {5, {5, 3, 1}},
        {6, {6, 3, 1}},
        {7, {7, 3, 1}},
        {8, {8, 5, 1}},
        {9, {9, 5, 1}},
        {10, {10, 6, 1}},
        {11, {11, 6, 1}},
        {12, {12, 7, 1}},
        {13, {13, 7, 1}},
        {14, {14, 8, 1}},
        {15, {15, 9, 1}},
    };

    return mg_level_lookup.at(N);
  }

  std::map<int, std::vector<int>> mg_level_lookup = {
      {1, {1}},
      {2, {2, 1}},
      {3, {3, 1}},
      {4, {4, 2, 1}},
      {5, {5, 3, 1}},
      {6, {6, 4, 2, 1}},
      {7, {7, 5, 3, 1}},
      {8, {8, 6, 4, 1}},
      {9, {9, 7, 5, 1}},
      {10, {10, 8, 5, 1}},
      {11, {11, 9, 5, 1}},
      {12, {12, 10, 5, 1}},
      {13, {13, 11, 5, 1}},
      {14, {14, 12, 5, 1}},
      {15, {15, 13, 5, 1}},
  };

  return mg_level_lookup.at(N);
}
