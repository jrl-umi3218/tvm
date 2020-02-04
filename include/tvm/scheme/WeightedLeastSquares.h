/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <tvm/scheme/abstract/ResolutionScheme.h>
#include <tvm/scheme/internal/Assignment.h>
#include <tvm/scheme/internal/ProblemComputationData.h>
#include <tvm/solver/abstract/LeastSquareSolver.h>

namespace tvm
{

namespace scheme
{
  /** This class implements the classic weighted least square scheme
    */
  class TVM_DLLAPI WeightedLeastSquares : public abstract::LinearResolutionScheme<WeightedLeastSquares>
  {
  private:
    struct Memory : public internal::ProblemComputationData
    {
      Memory(int solverId, std::unique_ptr<solver::abstract::LeastSquareSolver> solver);
      void resize(int m0, int m1, double big_number);

      std::unique_ptr<solver::abstract::LeastSquareSolver> solver;

    protected:
      void setVariablesToSolution_(VariableVector& x) override;
    };

    const static internal::SchemeAbilities abilities_;

  public:
    using ComputationDataType = Memory;

    // FIXME temporary verbose parameter
    //template<class SolverConfig, 
    //  typename std::enable_if_t<std::is_base_of<solver::abstract::LeastSquareConfiguration,SolverConfig>::value> = 0>
    template<class SolverConfig>
    WeightedLeastSquares(const SolverConfig& solverConfig,
                         bool verbose = true, double scalarizationWeight = 1000)
      : LinearResolutionScheme<WeightedLeastSquares>(abilities_)
      , verbose_(verbose)
      , scalarizationWeight_(scalarizationWeight)
      , solverConfig_(new SolverConfig(solverConfig))
    {
      static_assert(std::is_base_of<solver::abstract::LeastSquareConfiguration, SolverConfig>::value,
        "SolverConfig must derive from solver::abstract::LeastSquareConfiguration");
    }

    WeightedLeastSquares(const WeightedLeastSquares&) = delete;
    WeightedLeastSquares(WeightedLeastSquares&&) = delete;

    /** Private interface for CRTP*/
    bool solve_(LinearizedControlProblem& problem, internal::ProblemComputationData* data) const;
    std::unique_ptr<Memory> createComputationData_(const LinearizedControlProblem& problem) const;

  protected:
    bool verbose_;
    double scalarizationWeight_;
    std::unique_ptr<solver::abstract::LeastSquareConfiguration> solverConfig_;
  };

}  // namespace scheme

}  // namespace tvm
