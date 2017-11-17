#include <tvm/task_dynamics/Proportional.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{

  Proportional::Proportional(double kp)
    : kp_(kp)
  {
  }

  std::unique_ptr<abstract::TaskDynamicsImpl> Proportional::impl_(FunctionPtr f) const
  {
    return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, kp_));
  }

  Proportional::Impl::Impl(FunctionPtr f, double kp)
    : TaskDynamicsImpl(Order::One, f)
    , kp_(kp)
  {
  }

  void Proportional::Impl::updateValue()
  {
    value_ = -kp_ * function().value();
  }

}  // namespace task_dynamics

}  // namespace tvm
