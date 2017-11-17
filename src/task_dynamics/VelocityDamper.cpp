#include <tvm/task_dynamics/VelocityDamper.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{
  VelocityDamper::VelocityDamper(double xsi, double ds, double di)
    : dt_(0), xsi_(xsi), ds_(ds), di_(di)
  {
  }

  VelocityDamper::VelocityDamper(double dt, double xsi, double ds, double di)
    : dt_(dt), xsi_(xsi), ds_(ds), di_(di)
  {
    if (dt <= 0)
    {
      throw std::runtime_error("Time increment should be non-negative.");
    }
  }

  std::unique_ptr<abstract::TaskDynamicsImpl> VelocityDamper::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs) const
  {
    if (dt_ > 0)
    {
      return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, t, rhs, dt_, xsi_, ds_, di_));
    }
    else
    {
      return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, t, rhs, xsi_, ds_, di_));
    }
  }

  VelocityDamper::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, double xsi, double ds, double di)
    : TaskDynamicsImpl(Order::One, f, t, rhs)
    , dt_(0)
    , xsi_(xsi)
    , ds_(ds)
    , di_(di)
  {
    compute_ab();
  }

  VelocityDamper::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, double dt, double xsi, double ds, double di)
    : TaskDynamicsImpl(Order::Two, f, t, rhs)
    , dt_(dt)
    , xsi_(xsi)
    , ds_(ds)
    , di_(di)
  {
    compute_ab();
  }

  void VelocityDamper::Impl::updateValue()
  {
    value_ = a_ * (function().value() - rhs());
    value_.array() += b_;
    if (order() == Order::Two)
    {
      value_ -= function().velocity();
      value_ /= dt_;
    }
  }

  void VelocityDamper::Impl::compute_ab()
  {
    a_ = -xsi_ / (di_ - ds_);
    if (type() == constraint::Type::LOWER_THAN)
    {
      b_ = a_ * ds_;
    }
    else
    {
      b_ = - a_ * ds_;
    }
  }

} // task_dynamics

} // tvm
