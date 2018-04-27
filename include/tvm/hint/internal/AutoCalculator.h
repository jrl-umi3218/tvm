
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

#include <tvm/hint/abstract/SubstitutionCalculator.h>

namespace tvm
{

namespace hint
{

namespace internal
{

  /** Automatically generates the most appropriate calculator for the given
    * constraints and variables.
    *
    * Current rules:
    * - for a simple substitution with invertible diagonal matrix, generates a
    *   DiagonalCalculator
    * - otherwise generates a GenericCalculator
    *
    * \note You need to ensure that the matrix properties used when applying the
    * rules have been correctly set. If this is not the case, this should be
    * corrected at the function level. In particular it is improper to rely on
    * a run of the update pipeline to have all the properties correctly set.
    */
  class TVM_DLLAPI AutoCalculator : public abstract::SubstitutionCalculator
  {
  protected:
    std::unique_ptr<abstract::SubstitutionCalculatorImpl> impl_(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank) const;
  };

} // internal

} // hint

} // tvm