#include <tvm/task_dynamics/ProportionalDerivative.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{

  ProportionalDerivative::ProportionalDerivative(double kp, double kv)
    : TaskDynamics(Order::Two)
    , kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(double kp)
    : ProportionalDerivative(kp, std::sqrt(2*kp))
  {
  }

  void ProportionalDerivative::updateValue()
  {
    value_ = -kv_ * function().velocity() - kp_ * function().value();
  }

}  // namespace task_dynamics

}  // namespace tvm
