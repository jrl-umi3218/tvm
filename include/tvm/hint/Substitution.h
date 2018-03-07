#pragma once

/* Copyright 2017-2018 CNRS-UM LIRMM, CNRS-AIST JRL
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

#include <vector>

namespace tvm
{

namespace hint
{

  /** Hint for a substitution that could be done by the solver.*/
  class TVM_DLLAPI Substitution
  {
  public:
    Substitution(LinearConstraintPtr cstr, VariablePtr x, int rank = fullRank);
    Substitution(const std::vector<LinearConstraintPtr>& cstr, VariablePtr x, int rank = fullRank);
    Substitution(LinearConstraintPtr cstr, std::vector<VariablePtr>& x, int rank = fullRank);
    Substitution(const std::vector<LinearConstraintPtr>& cstr, std::vector<VariablePtr>& x, int rank = fullRank);

    int rank() const;
    const std::vector<LinearConstraintPtr>& constraints() const;
    const std::vector<VariablePtr>& variables() const;

    bool isSimple() const;

    /** Constant for specifying that the matrix in front of the variable is full
      * rank.
      */
    static const int fullRank = -1;

  private:
    void check() const;

    int rank_;
    std::vector<LinearConstraintPtr> constraints_;
    std::vector<VariablePtr> x_;
  };
}

}