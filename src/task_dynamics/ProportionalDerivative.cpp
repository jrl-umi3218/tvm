#include <tvm/task_dynamics/ProportionalDerivative.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{

  ProportionalDerivative::ProportionalDerivative(double kp, double kv)
    : kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(double kp)
    : ProportionalDerivative(kp, std::sqrt(2*kp))
  {
  }

  std::unique_ptr<abstract::TaskDynamicsImpl> ProportionalDerivative::impl_(FunctionPtr f) const
  {
    return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f,kp_,kv_));
  }

  ProportionalDerivative::Impl::Impl(FunctionPtr f, double kp, double kv)
    : TaskDynamicsImpl(Order::Two, f)
    , kp_(kp)
    , kv_(kv)
  {
  }

  void ProportionalDerivative::Impl::updateValue()
  {
    value_ = -kv_ * function().velocity() - kp_ * function().value();
  }

  std::pair<double, double> ProportionalDerivative::Impl::gains() const
  {
    return {kp_, kv_};
  }

  void ProportionalDerivative::Impl::gains(double kp, double kv)
  {
    kp_ = kp;
    kv_ = kv;
  }

  void ProportionalDerivative::Impl::gains(double kp)
  {
    kp_ = kp;
    kv_ = std::sqrt(2 * kp);
  }

}  // namespace task_dynamics

}  // namespace tvm
