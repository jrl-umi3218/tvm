/*
 * Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#include <tvm/task_dynamics/OneStepToZero.h>

#include <tvm/function/abstract/Function.h>

#include <sstream>

namespace tvm
{

namespace task_dynamics
{

void OneStepToZero::checkParam(Order d, double dt)
{
  if(d == Order::Zero)
    throw std::runtime_error("[task_dynamics::OneStepToZero] This task dynamics only work for order from 1. For order "
                             "0, maybe task_dynamics::None would fit you need.");

  if(d > Order::Two)
  {
    std::stringstream ss;
    ss << "[task_dynamics::OneStepToZero] Implementation only accepts order up to 2 (input was" << static_cast<int>(d)
       << ").";
    throw std::runtime_error(ss.str());
  }

  if(dt <= 0)
  {
    std::stringstream ss;
    ss << "[task_dynamics::OneStepToZero] dt needs to be positive (input was" << dt << ").";
    throw std::runtime_error(ss.str());
  }
}

OneStepToZero::OneStepToZero(Order d, double dt) : d_(d), dt_(dt) { checkParam(d, dt); }

std::unique_ptr<abstract::TaskDynamicsImpl> OneStepToZero::impl_(FunctionPtr f,
                                                                 constraint::Type t,
                                                                 const Eigen::VectorXd & rhs) const
{
  return std::make_unique<Impl>(f, t, rhs, d_, dt_);
}

Order OneStepToZero::order_() const { return d_; }

OneStepToZero::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, Order d, double dt)
: TaskDynamicsImpl(d, f, t, rhs), dt_(dt)
{
  OneStepToZero::checkParam(d, dt);
}

void OneStepToZero::Impl::updateValue()
{
  value_ = (rhs() - function().value()) / dt_;
  if(order() == Order::Two)
  {
    value_ -= function().velocity();
    value_ *= 2 / dt_;
  }
}

} // namespace task_dynamics

} // namespace tvm
