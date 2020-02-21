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

#include <tvm/VariableVector.h>
#include <tvm/scheme/internal/SchemeAbilities.h>
#include <tvm/scheme/internal/ProblemComputationData.h>
#include <tvm/scheme/internal/ResolutionSchemeBase.h>

namespace tvm
{

class LinearizedControlProblem;

namespace scheme
{

namespace abstract
{

  /** Resolution schemes may be defined only for a particular type of problems.
    * We use CRTP for providing a common interface despite this, and
    * performing some basic common operations.
    *
    * The Derived class must provide:
    * - one or several void solve_(ProblemType&, ComputationData&)
    *   methods (several if it handle differently several types of problems).
    * - likewise, one or several createComputationData_(const ProblemType&)
    *   methods returning a std::unique_ptr<ComputationDataType> where
    *   ComputationDataType derives from ProblemComputationData.
    *
    * For a given problem, the solve_ methods is guaranteed to receive the
    * ComputationData instance created by createComputationData_ for the same
    * problem.
    */
  template<typename Derived>
  class ResolutionScheme : public internal::ResolutionSchemeBase
  {
  public:
    template<typename Problem>
    bool solve(Problem& problem) const;

    template<typename Problem>
    std::unique_ptr<internal::ProblemComputationData> createComputationData(const Problem& problem) const;

    /** Returns a reference to the derived object */
    Derived& derived() { return *static_cast<Derived*>(this); }
    /** Returns a const reference to the derived object */
    const Derived& derived() const { return *static_cast<const Derived*>(this); }

  protected:
    ResolutionScheme(internal::SchemeAbilities abilities, double big = constant::big_number);
  };


  /** Base class for scheme solving linear problems
    * For now, it is there only for allowing to differentiate with future
    * non-linear schemes.
    */
  template<typename Derived>
  class LinearResolutionScheme : public ResolutionScheme<Derived>
  {
  protected:
    LinearResolutionScheme(internal::SchemeAbilities abilities, double big = constant::big_number);
  };



  template<typename Derived>
  template<typename Problem>
  inline bool ResolutionScheme<Derived>::solve(Problem& problem) const
  {
    auto data = getComputationData(problem, *this);
    problem.update();
    bool b = derived().solve_(problem, data);
    data->setVariablesToSolution();
    problem.substitutions().updateVariableValues();
    return b;
  }

  template<typename Derived>
  template<typename Problem>
  inline std::unique_ptr<internal::ProblemComputationData> ResolutionScheme<Derived>::createComputationData(const Problem& problem) const
  {
    return derived().createComputationData_(problem);
  }

  template<typename Derived>
  inline ResolutionScheme<Derived>::ResolutionScheme(internal::SchemeAbilities abilities, double big)
    :ResolutionSchemeBase(abilities, big)
  {
  }

  template<typename Derived>
  inline LinearResolutionScheme<Derived>::LinearResolutionScheme(internal::SchemeAbilities abilities, double big)
    : ResolutionScheme<Derived>(abilities, big)
  {
  }

}  // namespace abstract

}  // namespace scheme

}  // namespace tvm

#include <tvm/scheme/internal/helpers.hpp>
