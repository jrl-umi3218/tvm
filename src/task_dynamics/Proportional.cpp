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

  std::unique_ptr<abstract::TaskDynamicsImpl> Proportional::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
  {
    return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, t, rhs, kp_));
  }

  Proportional::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, double kp)
    : TaskDynamicsImpl(Order::One, f, t, rhs)
    , kp_(kp)
  {
  }

  void Proportional::Impl::updateValue()
  {
    value_ = -kp_ * (function().value() - rhs());
  }

  void Proportional::Impl::gain(double kp)
  {
    kp_ = kp;
  }

  double Proportional::Impl::gain() const
  {
    return kp_;
  }

}  // namespace task_dynamics

}  // namespace tvm
