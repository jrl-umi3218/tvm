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

#include <tvm/internal/FirstOrderProvider.h>

#include <Eigen/Core>

#include <memory>

namespace tvm
{

namespace constraint
{

namespace internal
{

  /** This is a helper class to define Constraint. Its sole purpose is to
   * declare the outputs L, U and E, (L and U being the lower and upper
   * bounds for inequality constraints, and E the term the constraint is
   * equal to for equality constraints), so that Constraint can the
   * dynamically disable what it does not use.
   */
  class TVM_DLLAPI ConstraintBase : public tvm::internal::FirstOrderProvider
  {
  public:
    SET_OUTPUTS(ConstraintBase, L, U, E)

  protected:
    ConstraintBase(int m);
  };

}  // namespace internal

}  // namespace constraint

}  // namespace tvm
