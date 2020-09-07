/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/function/IdentityFunction.h>

#include <tvm/Variable.h>

namespace tvm
{

namespace function
{

IdentityFunction::IdentityFunction(VariablePtr x)
: BasicLinearFunction(Eigen::MatrixXd::Identity(x->size(), x->size()), x)
{
  jacobian_.begin()->second.properties({tvm::internal::MatrixProperties::Shape::IDENTITY});
}

void IdentityFunction::A(const MatrixConstRef &, const Variable &, const tvm::internal::MatrixProperties &)
{
  throw std::runtime_error("You can not change the A matrix on a identity function");
}

void IdentityFunction::A(const MatrixConstRef &, const tvm::internal::MatrixProperties &)
{
  throw std::runtime_error("You can not change the A matrix on a identity function");
}

void IdentityFunction::b(const VectorConstRef &, const tvm::internal::MatrixProperties &)
{
  throw std::runtime_error("You can not change the b vector on a identity function");
}

void IdentityFunction::updateValue_() { value_ = variables()[0]->value(); }

void IdentityFunction::updateVelocity_() { velocity_ = dot(variables()[0])->value(); }

} // namespace function

} // namespace tvm
