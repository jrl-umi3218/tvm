/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{

namespace abstract
{

  std::unique_ptr<TaskDynamicsImpl> TaskDynamics::impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
  {
    auto ptr = impl_(f, t, rhs);
    return ptr;
  }

  Order TaskDynamics::order() const
  {
    return order_();
  }

}  // namespace abstract

}  // namespace task_dynamics

}  // namespace tvm
