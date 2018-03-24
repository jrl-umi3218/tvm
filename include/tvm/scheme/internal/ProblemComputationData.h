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

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/VariableVector.h>

#include <vector>

namespace tvm
{

namespace scheme
{

namespace internal
{

  /** Base class to be derived by each resolution scheme to hold all of the
    * temporary memories and states related to the resolution of a problem.
    * The rationale is that a resolution scheme is stateless, and when given
    * for the first time a problem to solve, creates a ProblemComputationData
    * that it gives to the problem and retrieves at each further resolution.
    */
  class TVM_DLLAPI ProblemComputationData
  {
  public:
    int solverId() const;

    void addVariable(VariablePtr var);
    void addVariable(const VariableVector& vars);
    void removeVariable(Variable* v);
    void removeVariable(const VariableVector& vars);
    const VariableVector& variables() const;

    /** Set the value of the variables. \a val must be in the same order as the
      * variables.
      */
    void setSolution(const VectorConstRef& val);

  protected:
    ProblemComputationData(int solverId);
    ProblemComputationData() = default;

    /** The problem variable*/
    VariableVector x_;

  private:
    int solverId_;
  };

  inline int ProblemComputationData::solverId() const
  {
    return solverId_;
  }

  inline void ProblemComputationData::addVariable(VariablePtr var)
  {
    x_.add(var);
  }

  inline void ProblemComputationData::addVariable(const VariableVector& vars)
  {
    x_.add(vars);
  }

  inline void ProblemComputationData::removeVariable(Variable* v)
  {
    //we don't raise an exception is the variable is not there, as we merge
    //identical variables when we add them.
    x_.remove(*v);
  }

  inline void ProblemComputationData::removeVariable(const VariableVector& vars)
  {
    for (const auto& v : vars.variables())
      removeVariable(v.get());
  }

  inline const VariableVector& ProblemComputationData::variables() const
  {
    return x_;
  }

  inline void ProblemComputationData::setSolution(const VectorConstRef & val)
  {
    x_.value(val);
  }

  inline ProblemComputationData::ProblemComputationData(int solverId)
    : solverId_(solverId)
  {
  }

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
