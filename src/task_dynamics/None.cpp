#include <tvm/task_dynamics/None.h>

#include <tvm/function/abstract/LinearFunction.h>

namespace tvm
{

  namespace task_dynamics
  {

    std::unique_ptr<abstract::TaskDynamicsImpl> None::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
    {
      return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, t, rhs));
    }

    None::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs)
      : TaskDynamicsImpl(Order::Zero, f, t, rhs)
    {
      using function::abstract::LinearFunction;
      lf_ = dynamic_cast<const LinearFunction*>(f.get());
      if (!lf_)
      {
        throw std::runtime_error("The function is not linear.");
      }
      addInputDependency<Impl>(Update::UpdateValue, std::static_pointer_cast<LinearFunction>(f), LinearFunction::Output::B);
    }

    void None::Impl::updateValue()
    {
      value_ = rhs() - lf_->b();
    }

  }  // namespace task_dynamics

}  // namespace tvm
