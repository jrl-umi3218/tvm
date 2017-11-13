#include <tvm/function/abstract/LinearFunction.h>

#include <tvm/Variable.h>

namespace tvm
{

namespace function
{

namespace abstract
{

  LinearFunction::LinearFunction(int m)
    :Function(m)
  {
    registerUpdates(LinearFunction::Update::Value, &LinearFunction::updateValue);
    registerUpdates(LinearFunction::Update::Velocity, &LinearFunction::updateVelocity);
    addOutputDependency<LinearFunction>(FirstOrderProvider::Output::Value, LinearFunction::Update::Value);
    addOutputDependency<LinearFunction>(Function::Output::Velocity, LinearFunction::Update::Velocity);
    setDerivativesToZero();
  }

  void LinearFunction::updateValue()
  {
    updateValue_();
  }

  void LinearFunction::updateVelocity()
  {
    updateVelocity_();
  }

  void LinearFunction::resizeCache()
  {
    Function::resizeCache();
    b_.resize(size());
    setDerivativesToZero();
  }

  void LinearFunction::updateValue_()
  {
    value_ = b_;
    for (auto v : variables())
      value_ += jacobian(*v) * v->value();
  }

  void LinearFunction::updateVelocity_()
  {
    value_.setZero();
    for (auto v : variables())
      value_ += jacobian(*v) * dot(v)->value();
  }

  void LinearFunction::setDerivativesToZero()
  {
    normalAcceleration_.setZero();
    for (const auto& v : variables())
      JDot_[v.get()].setZero();
  }

}  // namespace abstract

}  // namespace function

}  // namespace tvm
