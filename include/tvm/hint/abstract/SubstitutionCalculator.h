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
#include <tvm/hint/abstract/SubstitutionCalculatorImpl.h>

#include <memory>
#include <vector>

namespace tvm
{

namespace hint
{

namespace abstract
{
  /** A SubstitutionCalculator is a lightweight factory that can generates a
    * SubstitutionCalculatorImpl.
    * It is used to specify custom operations to be made during the substitution
    * process.
    */
  class TVM_DLLAPI SubstitutionCalculator
  {
  public:
    virtual ~SubstitutionCalculator() = default;

    std::unique_ptr<SubstitutionCalculatorImpl> impl(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank) const;

  protected:
    virtual std::unique_ptr<SubstitutionCalculatorImpl> impl_(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank) const = 0;

  };

} // internal

} // hint

} // tvm