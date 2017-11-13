#include <tvm/task_dynamics/None.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{

  None::None(const Eigen::VectorXd& v)
    : TaskDynamics(Order::Geometric)
  {
    value_ = v;
  }

  void None::updateValue()
  {
    // do nothing
  }

  void None::setFunction_()
  {
    if (value_.size() == 0)
      value_.setZero(function().size());
    else
      assert(value_.size() == function().size());
  }

}  // namespace task_dynamics

}  // namespace tvm
