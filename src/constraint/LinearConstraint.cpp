/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Variable.h>
#include <tvm/constraint/abstract/LinearConstraint.h>

namespace tvm
{

namespace constraint
{

namespace abstract
{

LinearConstraint::LinearConstraint(Type ct, RHS cr, int m) : Constraint(ct, cr, m)
{
  registerUpdates(LinearConstraint::Update::Value, &LinearConstraint::updateValue);
  addOutputDependency<LinearConstraint>(FirstOrderProvider::Output::Value, LinearConstraint::Update::Value);
}

void LinearConstraint::updateValue()
{
  value_.setZero();
  for(const auto & v : variables())
  {
    value_ += jacobian(*v) * v->value();
  }
}

} // namespace abstract

} // namespace constraint

} // namespace tvm
