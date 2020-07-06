/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

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

  Proportional::Proportional(const Eigen::VectorXd& kp)
    : kp_(kp)
  {
  }

  Proportional::Proportional(const  Eigen::MatrixXd& kp)
    : kp_(kp)
  {
  }

  std::unique_ptr<abstract::TaskDynamicsImpl> Proportional::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
  {
    return std::make_unique<Impl>(f, t, rhs, kp_);
  }

  Order Proportional::order_() const
  {
    return Order::One;
  }

  Proportional::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, const Gain& kp)
    : TaskDynamicsImpl(Order::One, f, t, rhs)
    , kp_(kp)
  {
    if ((kp.index() == 1 && mpark::get<Eigen::VectorXd>(kp).size() != f->size())   // Diagonal gain
     || (kp.index() == 2 && mpark::get<Eigen::MatrixXd>(kp).cols() != f->size()    // Matrix gain
                         && mpark::get<Eigen::MatrixXd>(kp).rows() != f->size()))
    {
      throw std::runtime_error("[task_dynamics::Proportional::Impl] Gain and function have incompatible sizes.");
    }
  }

  void Proportional::Impl::updateValue()
  {
    // \internal The code below would be cleaner using std::visit. Unfortunately
    // this is slower because it prevents Eigen to perform some optimization.
    switch (kp_.index())
    {
    case 0: value_ = -mpark::get<double>(kp_) * (function().value() - rhs()); break;
    case 1: value_.noalias() = -(mpark::get<Eigen::VectorXd>(kp_).asDiagonal() * (function().value() - rhs())); break;
    case 2: value_.noalias() = -mpark::get<Eigen::MatrixXd>(kp_) * (function().value() - rhs()); break;
    default: assert(false);
    }
  }

  void Proportional::Impl::gain(double kp)
  {
    kp_ = kp;
  }

  void Proportional::Impl::gain(const Eigen::VectorXd& kp)
  {
    if (kp.size() != function().size())
    {
      throw std::runtime_error("[task_dynamics::Proportional::Impl::gain] Gain and function have incompatible sizes.");
    }
    kp_ = kp;
  }

  void Proportional::Impl::gain(const Eigen::MatrixXd& kp)
  {
    if (kp.rows() != function().size() || kp.cols() != function().size())
    {
      throw std::runtime_error("[task_dynamics::Proportional::Impl::gain] Gain and function have incompatible sizes.");
    }
    kp_ = kp;
  }

  const Proportional::Gain& Proportional::Impl::gain() const
  {
    return kp_;
  }

  Proportional::Gain& Proportional::Impl::gain()
  {
    return kp_;
  }

}  // namespace task_dynamics

}  // namespace tvm
