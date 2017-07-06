#pragma once

namespace tvm
{
  /** For a function f(x), and a right hand side rhs (and rhs2):
  * EQUAL        f(x) =  rhs
  * GREATER_THAN f(x) >= rhs
  * LOWER_THAN   f(x) <= rhs
  * DOUBLE_SIDED rhs <= f(x) <= rhs2
  */
  enum class ConstraintType
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
  enum class RHSType
  {
    ZERO,
    AS_GIVEN,
    OPPOSITE
  };
}
