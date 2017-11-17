#include <tvm/task_dynamics/Proportional.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{

  Proportional::Proportional(double kp)
    : TaskDynamics(Order::One)
    , kp_(kp)
  {
  }

  void Proportional::updateValue()
  {
    value_ = -kp_ * function().value();
  }

}  // namespace task_dynamics

}  // namespace tvm
