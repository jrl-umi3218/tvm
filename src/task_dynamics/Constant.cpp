/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#include <tvm/task_dynamics/Constant.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

  namespace task_dynamics
  {

    Constant::Constant()
    {
    }

    std::unique_ptr<abstract::TaskDynamicsImpl> Constant::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
    {
      return std::make_unique<Impl>(f, t, rhs);
    }

    Order Constant::order_() const
    {
      return Order::Zero;
    }

    Constant::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs)
      : TaskDynamicsImpl(Order::Zero, f, t, rhs)
    {
      value_ = rhs;
    }

    void Constant::Impl::updateValue()
    {
      value_ = rhs();
    }

  }  // namespace task_dynamics

}  // namespace tvm
