#pragma once

#include <initializer_list>
#include <memory>

#include <Eigen/Core>

#include "Function.h"
#include "defs.h"


namespace tvm
{
  class TVM_DLLAPI LinearFunction : public Function
  {
  public:
    SET_UPDATES(LinearFunction, Value, Velocity)

    void updateValue();
    void updateVelocity();
    void resizeCache() override;

  protected:
    LinearFunction(int m);
    virtual void updateValue_();
    virtual void updateVelocity_();
    void setDerivativesToZero();

    Eigen::VectorXd b_;
  };


  /** The most basic linear function f(x_1, ..., x_k) = sum A_i x_i + b where
    * the matrices are constant.
    */
  class TVM_DLLAPI BasicLinearFunction : public LinearFunction
  {
  public:
    /** b = 0 */
    BasicLinearFunction(const MatrixConstRef& A, VariablePtr x);
    BasicLinearFunction(std::initializer_list<MatrixConstRef> A, std::initializer_list<VariablePtr> x);

    /** b is user-supplied*/
    BasicLinearFunction(const MatrixConstRef& A, VariablePtr x, const VectorConstRef& b);
    BasicLinearFunction(std::initializer_list<MatrixConstRef> A, std::initializer_list<VariablePtr> x, const VectorConstRef& b);

    /** Set the matrix A corresponding to variable x.*/
    virtual void setA(const MatrixConstRef& A, const Variable& x);
    /** Shortcut for when there is a single variable.*/
    virtual void setA(const MatrixConstRef& A);
    /** Set the constant term b.*/
    virtual void setb(const VectorConstRef& b);

  private:
    void add(const Eigen::MatrixXd& A, VariablePtr x);
  };

  /** f(x) = x, for a single variable.*/
  class TVM_DLLAPI IdentityFunction : public BasicLinearFunction
  {
  public:
    IdentityFunction(VariablePtr x);

    void setA(const MatrixConstRef& A, const Variable& x) override;
    void setA(const MatrixConstRef& A) override;
    void setb(const VectorConstRef& b) override;

  protected:
    void updateValue_() override;
    void updateVelocity_() override;
  };
}
