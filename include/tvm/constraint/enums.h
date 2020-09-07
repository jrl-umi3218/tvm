/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

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
