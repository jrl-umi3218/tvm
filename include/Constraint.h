#pragma once

#include "FirstOrderProvider.h"

#include <Eigen/Core>

#include <memory>

namespace taskvm
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

  class Variable;

  class Constraint : public internal::FirstOrderProvider
  {
  public:
    /** Note: I don't like the denomination rhs very much. 
      * It would be cleare to have lowerBound() and upperBound() (or l()/u()), 
      * but this would require to test when one or the other is valid, depending 
      * on the constraint type.
      * With rhs and rhs2, we still have the problem that one or both might be
      * undefined.
      */
    const Eigen::VectorXd& l() const;
    const Eigen::VectorXd& u() const;

  protected:
    void resizeCache() override;

    Eigen::VectorXd& l();
    Eigen::VectorXd& u();

  private:
    ConstraintType  cstrType_;
    RHSType         rhsType_;

    Eigen::VectorXd l_;
    Eigen::VectorXd u_;
  };
}