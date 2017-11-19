#include <tvm/task_dynamics/VelocityDamper.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{
  VelocityDamper::VelocityDamper(double xsi, double ds, double di, double big)
    : dt_(0), xsi_(xsi), ds_(ds), di_(di), big_(big)
  {
  }

  VelocityDamper::VelocityDamper(double dt, double xsi, double ds, double di, double big)
    : dt_(dt), xsi_(xsi), ds_(ds), di_(di), big_(big)
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
      return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, t, rhs, dt_, xsi_, ds_, di_, big_));
    }
    else
    {
      return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, t, rhs, xsi_, ds_, di_, big_));
    }
  }

  VelocityDamper::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, double xsi, double ds, double di, double big)
    : TaskDynamicsImpl(Order::One, f, t, rhs)
    , dt_(0)
    , xsi_(xsi)
    , ds_(ds)
    , di_(di)
    , big_(big)
  {
    compute_ab();
    d_.resize(f->size());
  }

  VelocityDamper::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, double dt, double xsi, double ds, double di, double big)
    : TaskDynamicsImpl(Order::Two, f, t, rhs)
    , dt_(dt)
    , xsi_(xsi)
    , ds_(ds)
    , di_(di)
    , big_(big)
  {
    compute_ab();
    d_.resize(f->size());
  }

  void VelocityDamper::Impl::updateValue()
  {
    d_ = (function().value() - rhs());
    value_ = a_ * d_;
    value_.array() += b_;
    if (order() == Order::Two)
    {
      value_ -= function().velocity();
      value_ /= dt_;
    }

    //if distance to constraint greater than di, we "deactivate" the constraint by puting the value to big
    if (type() == constraint::Type::LOWER_THAN)
    {
      value_ = (d_.array() <= -di_).select(value_, big_);
    }
    else
    {
      value_ = (d_.array() >= di_).select(value_, -big_);
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
