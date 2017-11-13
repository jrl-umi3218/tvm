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

  /** For a function f(x), and a right hand side rhs (and rhs2):
    * EQUAL        f(x) =  rhs
    * GREATER_THAN f(x) >= rhs
    * LOWER_THAN   f(x) <= rhs
    * DOUBLE_SIDED rhs <= f(x) <= rhs2
    */
  enum class Type
  {
    EQUAL,
    GREATER_THAN,
    LOWER_THAN,
    DOUBLE_SIDED
  };

  /** Given a vector u:
    * ZERO      rhs = 0
    * AS_GIVEN  rhs = u
    * OPPOSITE  rhs = -u
    */
  enum class RHS
  {
    ZERO,
    AS_GIVEN,
    OPPOSITE
  };

} // namespace constraint

} // namespace tvm
