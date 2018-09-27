#pragma once

/* Copyright 2018 CNRS-UM LIRMM, CNRS-AIST JRL
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
#include <tvm/scheme/internal/Assignment.h>

#include <map>

namespace tvm
{

namespace scheme
{

namespace abstract
{
  /** Base class for a (constrained) least-square solver. */
  class TVM_DLLAPI LeastSquareSolver
  {
  public:
    LeastSquareSolver(const LeastSquareSolver&) = delete;
    LeastSquareSolver& operator=(const LeastSquareSolver&) = delete;
    /** Preallocate the memory
      */
    void reserve(const VariableVector& x, int m0, int me, int mi, bool useBounds=true);
    void addBound(LinearConstraintPtr bound);
    void addObjective();

    /** Set ||x||^2 as the least square objective of the problem.
      * \warning this replace previously added objectives.
      */
    void setMinimumNorm();
    bool solve();

  protected:
    virtual void reserve_(int m0, int me, int mi, bool useBounds) = 0;
    virtual void addBound_(LinearConstraintPtr bound, RangePtr range, bool first) = 0;
    virtual void addEqualityConstraint();
    virtual void addIneqalityConstraint();

    std::vector<internal::Assignment> assignments_;

  private:
    VariableVector const *x_;
    std::map<Variable*, bool> first_;  // For bound assignment
  };
}

}

}