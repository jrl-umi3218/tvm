/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/function/abstract/LinearFunction.h>

#include <tvm/Variable.h>

namespace tvm
{

namespace function
{

namespace abstract
{

LinearFunction::LinearFunction(int m) : Function(m), b_(m)
{
  registerUpdates(Update::Value, &LinearFunction::updateValue);
  registerUpdates(Update::Velocity, &LinearFunction::updateVelocity);
  addInputDependency<LinearFunction>(Update::Value, *this, Output::Jacobian);
  addInputDependency<LinearFunction>(Update::Value, *this, Output::B);
  addInputDependency<LinearFunction>(Update::Velocity, *this, Output::Jacobian);
  addOutputDependency<LinearFunction>(Output::Value, Update::Value);
  addOutputDependency<LinearFunction>(Output::Velocity, Update::Velocity);
}

void LinearFunction::updateValue() { updateValue_(); }

void LinearFunction::updateVelocity() { updateVelocity_(); }

void LinearFunction::resizeCache()
{
  Function::resizeCache();
  b_.resize(size());
  setDerivativesToZero();
}

void LinearFunction::updateValue_()
{
  value_ = b_;
  for(auto v : variables())
    value_ += jacobian(*v) * v->value();
}

void LinearFunction::updateVelocity_()
{
  velocity_.setZero();
  for(auto v : variables())
    velocity_ += jacobian(*v) * dot(v)->value();
}

void LinearFunction::setDerivativesToZero()
{
  normalAcceleration_.setZero();
  for(const auto & v : variables())
    JDot_[v.get()].setZero();
}

} // namespace abstract

} // namespace function

} // namespace tvm
