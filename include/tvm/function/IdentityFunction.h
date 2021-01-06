/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/function/BasicLinearFunction.h>

namespace tvm
{

namespace function
{

/** f(x) = x, for a single variable.*/
class TVM_DLLAPI IdentityFunction : public BasicLinearFunction
{
public:
  /** Build an identity function on variable \p x*/
  IdentityFunction(VariablePtr x);

protected:
  void updateValue_() override;
  void updateVelocity_() override;

private:
  /** Overridden function that always throws.*/
  void A(const MatrixConstRef & A,
         const Variable & x,
         const tvm::internal::MatrixProperties & p = tvm::internal::MatrixProperties()) override;
  /** Overridden function that always throws.*/
  void A(const MatrixConstRef & A,
         const tvm::internal::MatrixProperties & p = tvm::internal::MatrixProperties()) override;
  /** Overridden function that always throws.*/
  void b(const VectorConstRef & b, const tvm::internal::MatrixProperties &) override;
};

} // namespace function

} // namespace tvm
