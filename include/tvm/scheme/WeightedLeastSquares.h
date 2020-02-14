/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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

#include <tvm/internal/meta.h>
#include <tvm/scheme/abstract/ResolutionScheme.h>
#include <tvm/scheme/internal/Assignment.h>
#include <tvm/scheme/internal/ProblemComputationData.h>
#include <tvm/solver/abstract/LeastSquareSolver.h>

// Creating a class has_member_type_Config<T>
CREATE_HAS_MEMBER_TYPE_TRAIT_FOR(Config)


namespace tvm
{

namespace scheme
{
  /** This class implements the classic weighted least square scheme. */
  class TVM_DLLAPI WeightedLeastSquares : public abstract::LinearResolutionScheme<WeightedLeastSquares>
  {
  private:
    struct Memory : public internal::ProblemComputationData
    {
      Memory(int solverId, std::unique_ptr<solver::abstract::LeastSquareSolver> solver);

      std::unique_ptr<solver::abstract::LeastSquareSolver> solver;

    protected:
      void setVariablesToSolution_(VariableVector& x) override;
    };

    const static internal::SchemeAbilities abilities_;

    /** Check if T derives from LeastSquareSolverConfiguration. */
    template<typename T> using isConfig = std::is_base_of<solver::abstract::LeastSquareConfiguration, T>;
    /** Helper struct for isOption .*/
    template<typename T, bool> struct isOption_ : std::false_type {};
    /** Helper struct specialization for isOption .*/
    template<typename T> struct isOption_<T, true> { static const bool value = isConfig<typename T::Config>::value;  };
    /** Check if T has a member T::Config and if so if T::Config derives from LeastSquareSolverConfiguration.*/
    template<typename T> using isOption = isOption_<T, has_member_type_Config<T>::value>;

  public:
    using ComputationDataType = Memory;

    /** Constructor from a LeastSquareConfiguration
      * \tparam SolverConfig Any class deriving from LeastSquareConfiguration.
      * \param solverConfig A configuration for the solver to be used by the resolution scheme.
      * \param scalarizationWeight The factor to emulate priority for priority levels >= 1.
      *    E.g. if a task T1 has a weight w1 and priority 1, and a task T2 has a weight w2 and
      *    priority 2, the weighted least-squares problem will be assembled with weights
      *    1000*w1 and w2 for T1 and T2 respectively.
      */
    template<class SolverConfig, 
      typename std::enable_if<isConfig<SolverConfig>::value, int>::type = 0>
    WeightedLeastSquares(const SolverConfig& solverConfig, double scalarizationWeight = 1000)
      : LinearResolutionScheme<WeightedLeastSquares>(abilities_)
      , scalarizationWeight_(scalarizationWeight)
      , solverConfig_(solverConfig.clone())
    {
    }

    /** Constructor from a configuration class
      * \tparam SolverOptions Any class representing solver options. The class must have a
      *    member type \a Config refering to a class C deriving from LeastSquareConfiguration
      *    and such that C can be constructed from SolverOptions.
      * \param solverOptions A set of options for the solver to be used by the resolution scheme.
      * \param scalarizationWeight The factor to emulate priority for priority levels >= 1.
      *    E.g. if a task T1 has a weight w1 and priority 1, and a task T2 has a weight w2 and
      *    priority 2, the weighted least-squares problem will be assembled with weights
      *    1000*w1 and w2 for T1 and T2 respectively.
      */
    template<class SolverOptions,
      typename std::enable_if<isOption<SolverOptions>::value, int>::type = 0>
    WeightedLeastSquares(const SolverOptions& solverOptions, double scalarizationWeight = 1000)
      :WeightedLeastSquares(typename SolverOptions::Config(solverOptions), scalarizationWeight)
    {
    }

    /** A fallback constructor that is enable when none of the others are. 
      * It always fails at compilation time to provide a nice error message.
      */
    template<typename T, 
      typename std::enable_if<!isConfig<T>::value && !isOption<T>::value, int>::type = 0>
    WeightedLeastSquares(const T&, double = 1000)
      : LinearResolutionScheme<WeightedLeastSquares>(abilities_)
    {
      static_assert(tvm::internal::always_false<T>::value, 
        "First argument can only be a LeastSquareConfiguration or a solver configuration. "
        "A configuration needs to have a Config member type that is itself deriving from LeastSquareConfiguration. "
        "See LSSOLLeastSquareOptions for an example.");
    }


    WeightedLeastSquares(const WeightedLeastSquares&) = delete;
    WeightedLeastSquares(WeightedLeastSquares&&) = delete;

    /** Private interface for CRTP*/
    bool solve_(LinearizedControlProblem& problem, internal::ProblemComputationData* data) const;
    std::unique_ptr<Memory> createComputationData_(const LinearizedControlProblem& problem) const;

  protected:
    double scalarizationWeight_;
    /** The factory to create solvers attached to each problem. */
    std::unique_ptr<solver::abstract::LeastSquareConfiguration> solverConfig_;
  };

}  // namespace scheme

}  // namespace tvm
