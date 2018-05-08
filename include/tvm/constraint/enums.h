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

namespace tvm
{

namespace constraint
{

  /** For a function f(x), and a right hand side rhs (and rhs2), gives the type
    * of the constraint.
    */
  enum class Type
  {
    /** f(x) =  rhs */
    EQUAL,
    /** f(x) >= rhs */
    GREATER_THAN,
    /** f(x) <= rhs */
    LOWER_THAN,
    /** rhs <= f(x) <= rhs2 */
    DOUBLE_SIDED
  };

  /** Tell how a vector \p u should be considered as a right hand side of a
    * constraint.
    */
  enum class RHS
  {
    /** rhs = 0 */
    ZERO,
    /** rhs = u */
    AS_GIVEN,
    /** rhs = -u */
    OPPOSITE
  };

} // namespace constraint

} // namespace tvm
