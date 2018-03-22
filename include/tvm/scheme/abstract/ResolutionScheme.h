#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

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
    * - a typedef ComputationDataType defining the type of the
    *   ProblemComputationData it uses,
    * - one or several void solve_(ProblemType&, ComputationDataType&)
    *   methods (several if it handle differently several types of problems).
    * - likewise, one or several createComputationData_(const ProblemType&)
    *   methods returning a std::unique_ptr<ComputationDataType>.
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
    //We assume here that the resolution scheme has only one type of computation data even if
    //it can discriminate between several type of problems.
    //Should it not be the case, we could use traits to determine the ComputationDataType for
    //a Problem, given a particular ResolutionScheme
    auto& data = static_cast<typename Derived::ComputationDataType&>(getComputationData(problem, *this));
    problem.update();
    return derived().solve_(problem, data);
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
