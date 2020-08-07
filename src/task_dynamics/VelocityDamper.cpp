/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#include <tvm/task_dynamics/VelocityDamper.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{
  VelocityDamper::Config::Config(double di, double ds, double xsi, double xsiOff)
    : di_(di), ds_(ds), xsi_(xsi), xsiOff_(xsiOff)
  {
    if (di_ <= ds_)
    {
      throw std::runtime_error("di must be greater than ds");
    }
    if (ds_ < 0)
    {
      throw std::runtime_error("ds should be non-negative.");
    }
    if (xsi_ < 0)
    {
      throw std::runtime_error("xsi should be non-negative.");
    }
    if (xsiOff_ < 0)
    {
      throw std::runtime_error("xsiOff should be positive.");
    }
  }

  VelocityDamper::AnisotropicConfig::AnisotropicConfig(const VectorConstRef & di,
                                                       const VectorConstRef & ds,
                                                       const VectorConstRef & xsi,
                                                       const std::optional<VectorConstRef> & xsiOff)
  : di_(di), ds_(ds), xsi_(xsi),
    xsiOff_(xsiOff.value_or(Eigen::VectorXd::Constant(di_.size(), 1, 0.0)))
  {
    if(di_.size() != ds_.size() || di_.size() != xsi_.size() || di_.size() != xsiOff_.size())
    {
      throw std::runtime_error("All vector components must have the same size");
    }
    if((di_.array() <= ds_.array()).any())
    {
      throw std::runtime_error("di must be greater than ds");
    }
    if((ds_.array() < 0).any())
    {
      throw std::runtime_error("ds should be non-negative.");
    }
    if((xsi_.array() < 0).any())
    {
      throw std::runtime_error("xsi should be non-negative.");
    }
    auto automatic = xsi_.array() == 0;
    if(automatic.any() && !automatic.all())
    {
      throw std::runtime_error("Automatic damping must be enabled for all dimensions");
    }
    if ((xsiOff_.array() < 0).any())
    {
      throw std::runtime_error("xsiOff should be positive.");
    }
  }

  VelocityDamper::AnisotropicConfig::AnisotropicConfig(const Config & config)
  : AnisotropicConfig(Eigen::VectorXd::Constant(1, 1, config.di_),
                      Eigen::VectorXd::Constant(1, 1, config.ds_),
                      Eigen::VectorXd::Constant(1, 1, config.xsi_),
                      Eigen::VectorXd::Constant(1, 1, config.xsiOff_))
  {
  }

  VelocityDamper::VelocityDamper(const Config & config, double big)
  : VelocityDamper(AnisotropicConfig{config}, big)
  {
  }

  VelocityDamper::VelocityDamper(const AnisotropicConfig& config, double big)
    : dt_(0)
    , xsi_(0)
    , ds_(config.ds_)
    , di_(config.di_)
    , big_(big)
    , autoXsi_(config.xsi_(0) == 0) // we have ensure they are either all 0 or all specified
  {
    if (autoXsi_)
    {
      xsi_ = config.xsiOff_;
    }
    else
    {
      xsi_ = config.xsi_;
    }
    if (big <= 0)
    {
      throw std::runtime_error("big should be non-negative.");
    }
  }

  VelocityDamper::VelocityDamper(double dt, const Config & config, double big)
  : VelocityDamper(dt, AnisotropicConfig{config}, big)
  {
  }

  VelocityDamper::VelocityDamper(double dt, const AnisotropicConfig& config, double big)
    : dt_(dt)
    , xsi_(0)
    , ds_(config.ds_)
    , di_(config.di_)
    , big_(big)
    , autoXsi_(config.xsi_(0) == 0) // we have ensure they are either all 0 or all specified
  {
    if (autoXsi_)
    {
      xsi_ = config.xsiOff_;
    }
    else
    {
      xsi_ = config.xsi_;
    }
    if (dt <= 0)
    {
      throw std::runtime_error("Time increment should be non-negative.");
    }
    if (big <= 0)
    {
      throw std::runtime_error("big should be non-negative.");
    }
  }

  std::unique_ptr<abstract::TaskDynamicsImpl> VelocityDamper::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs) const
  {
    if (dt_ > 0)
    {
      return std::make_unique<Impl>(f, t, rhs, dt_, autoXsi_, di_, ds_, xsi_, big_);
    }
    else
    {
      return std::make_unique<Impl>(f, t, rhs, autoXsi_, di_, ds_, xsi_, big_);
    }
  }

  Order VelocityDamper::order_() const
  {
    return dt_ > 0 ? Order::Two : Order::One;
  }

  namespace
  {

  static inline Eigen::VectorXd resizeParameter(const FunctionPtr & f, const Eigen::VectorXd & in, const char * desc)
  {
    if(in.size() == 1)
    {
      return Eigen::VectorXd::Constant(f->size(), 1, in(0));
    }
    else if(in.size() != f->size())
    {
      std::string error = "Size of the " + std::string(desc) + " vector ( " + std::to_string(in.size()) + " ) does not match the function size ( " + std::to_string(f->size()) + " ) ";
      throw std::runtime_error(error.c_str());
    }
    return in;
  }

  }

  VelocityDamper::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, bool autoXsi, const Eigen::VectorXd & di, const Eigen::VectorXd & ds, const Eigen::VectorXd & xsi, double big)
    : TaskDynamicsImpl(Order::One, f, t, rhs)
    , dt_(0)
    , ds_(resizeParameter(f, ds, "safety distance"))
    , di_(resizeParameter(f, di, "interaction distance"))
    , xsiOff_(0)
    , a_(-(di_ - ds_).cwiseInverse())
    , big_(big)
    , autoXsi_(autoXsi)
    , d_(f->size())
    , axsi_(f->size())
    , active_(f->size(), false)
  {
    if (autoXsi)
    {
      axsi_.setOnes();
      xsiOff_ = resizeParameter(f, xsi, "damping offset parameter");
      addInputDependency<TaskDynamicsImpl>(Update::UpdateValue, f, function::abstract::Function::Output::Velocity);
    }
    else
    {
      axsi_ = a_.array() * resizeParameter(f, xsi, "damping parameter").array();
    }
  }

  VelocityDamper::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, double dt, bool autoXsi, const Eigen::VectorXd & di, const Eigen::VectorXd & ds, const Eigen::VectorXd & xsi, double big)
    : TaskDynamicsImpl(Order::Two, f, t, rhs)
    , dt_(dt)
    , ds_(resizeParameter(f, ds, "safety distance"))
    , di_(resizeParameter(f, di, "interaction distance"))
    , xsiOff_(0)
    , a_(-(di_ - ds_).cwiseInverse())
    , big_(big)
    , autoXsi_(autoXsi)
    , d_(f->size())
    , axsi_(f->size())
    , active_(f->size(), false)
  {
    if (autoXsi)
    {
      axsi_.setOnes();
      xsiOff_ = resizeParameter(f, xsi, "damping offset parameter");
    }
    else
    {
      axsi_ = a_.array() * resizeParameter(f, xsi, "damping parameter").array();
    }
  }

  void VelocityDamper::Impl::updateValue()
  {
    if (type() == constraint::Type::LOWER_THAN)
    {
      d_ = rhs() - function().value(); //turn f<=rhs into d = rhs-f >= 0
      updateValue_(-1);
      if (order() == Order::Two)
      {
        value_ += function().velocity();
        value_ /= dt_;
      }
      value_ = (d_.array() > di_.array()).select(big_, -value_);
    }
    else
    {
      d_ = function().value() - rhs(); //turn f>=rhs into d = f-rhs >= 0
      updateValue_(+1);
      if (order() == Order::Two)
      {
        value_ -= function().velocity();
        value_ /= dt_;
      }
      value_ = (d_.array() > di_.array()).select(-big_, value_);
    }
  }

  void VelocityDamper::Impl::updateValue_(double s)
  {
    if (autoXsi_)
    {
      const auto& dv = function().velocity();
      for (int i = 0; i < function().size(); ++i)
      {
        if (d_[i] <= di_(i) && !active_[static_cast<size_t>(i)])
        {
          active_[static_cast<size_t>(i)] = true;
          axsi_[i] = a_(i) * (s * dv[i] * (ds_(i) - di_(i)) / (d_[i] - ds_(i)) + xsiOff_(i));
        }
        else if (d_[i] > di_(i) && active_[static_cast<size_t>(i)])
        {
          active_[static_cast<size_t>(i)] = false;
        }
      }
    }
    value_.array() = d_.array() - ds_.array();
    value_.array() *= axsi_.array();
  }

} // task_dynamics

} // tvm
