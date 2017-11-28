#include <tvm/task_dynamics/Constant.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

  namespace task_dynamics
  {

    Constant::Constant()
    {
    }

    std::unique_ptr<abstract::TaskDynamicsImpl> Constant::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
    {
      return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, t, rhs));
    }

    Constant::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs)
      : TaskDynamicsImpl(Order::Zero, f, t, rhs)
    {
      value_ = rhs;
    }

    void Constant::Impl::updateValue()
    {
      // do nothing
    }

  }  // namespace task_dynamics

}  // namespace tvm
