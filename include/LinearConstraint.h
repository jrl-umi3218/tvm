#pragma once

#include <initializer_list>
#include <memory>

#include <Eigen/Core>

#include "Constraint.h"
#include "defs.h"


namespace tvm
{
  class Variable;

  class TVM_DLLAPI LinearConstraint : public Constraint
  {
  public:
    SET_UPDATES(LinearConstraint, Value);

    void updateValue();

  protected:
    LinearConstraint(ConstraintType ct, RHSType rt, int m);
  };


  /** The most basic linear constraint where the matrix and the vector(s) are
    * constant.
    */
  class TVM_DLLAPI BasicLinearConstraint : public LinearConstraint
  {
  public:
    /** Ax = 0, Ax <= 0 or Ax >= 0. */
    BasicLinearConstraint(const MatrixConstRef& A, std::shared_ptr<Variable> x, ConstraintType ct);
    BasicLinearConstraint(std::initializer_list<MatrixConstRef> A, std::initializer_list<std::shared_ptr<Variable>> x, ConstraintType ct);
    /** Ax = +/-b, Ax <= +/-b or Ax >= +/-b */
    BasicLinearConstraint(const MatrixConstRef& A, std::shared_ptr<Variable> x, const VectorConstRef& b,
                          ConstraintType ct, RHSType rt = RHSType::AS_GIVEN);
    BasicLinearConstraint(std::initializer_list<MatrixConstRef> A, std::initializer_list<std::shared_ptr<Variable>> x, const VectorConstRef& b,
                          ConstraintType ct, RHSType rt = RHSType::AS_GIVEN);
    /** l <= Ax <= u */
    BasicLinearConstraint(const MatrixConstRef& A, std::shared_ptr<Variable> x,
                          const VectorConstRef& l, const VectorConstRef& u, RHSType rt = RHSType::AS_GIVEN);
    BasicLinearConstraint(std::initializer_list<MatrixConstRef> A, std::initializer_list<std::shared_ptr<Variable>> x,
                          const VectorConstRef& l, const VectorConstRef& u, RHSType rt = RHSType::AS_GIVEN);

    /** Set the matrix A corresponding to variable x.*/
    void setA(const MatrixConstRef& A, const Variable& x);
    /** Shortcut for when there is a single variable.*/
    void setA(const MatrixConstRef& A);
    void setb(const VectorConstRef& b);
    void setl(const VectorConstRef& l);
    void setu(const VectorConstRef& l);

  private:
    void add(const Eigen::MatrixXd& A, std::shared_ptr<Variable> x);
  };
}
