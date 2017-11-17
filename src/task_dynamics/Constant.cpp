#include <tvm/task_dynamics/Constant.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

  namespace task_dynamics
  {

    Constant::Constant(const Eigen::VectorXd& v)
      : TaskDynamics(Order::Zero)
    {
      value_ = v;
    }

    void Constant::updateValue()
    {
      // do nothing
    }

    void Constant::setFunction_()
    {
      if (value_.size() == 0)
        value_.setZero(function().size());
      else
        assert(value_.size() == function().size());
    }

  }  // namespace task_dynamics

}  // namespace tvm
