#include <tvm/task_dynamics/Constant.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

  namespace task_dynamics
  {

    Constant::Constant(const Eigen::VectorXd& v)
      :v_(v)
    {
    }

    std::unique_ptr<abstract::TaskDynamicsImpl> Constant::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
    {
      return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, t, rhs, v_));
    }

    Constant::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, const Eigen::VectorXd& v)
      : TaskDynamicsImpl(Order::Zero, f, t, rhs)
    {
      if (value_.size() == 0)
        value_.setZero(function().size());
      else
      {
        assert(value_.size() == function().size());
        value_ = v;
      }
    }

    void Constant::Impl::updateValue()
    {
      // do nothing
    }

  }  // namespace task_dynamics

}  // namespace tvm
