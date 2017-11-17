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

    std::unique_ptr<abstract::TaskDynamicsImpl> Constant::impl_(FunctionPtr f) const
    {
      return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, v_));
    }

    Constant::Impl::Impl(FunctionPtr f, const Eigen::VectorXd& v)
      : TaskDynamicsImpl(Order::Zero, f)
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
