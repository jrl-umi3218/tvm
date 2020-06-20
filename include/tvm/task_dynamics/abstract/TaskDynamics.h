/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#pragma once

#include <tvm/defs.h>
#include <tvm/graph/abstract/Node.h>
#include <tvm/task_dynamics/enums.h>
#include <tvm/task_dynamics/abstract/TaskDynamicsImpl.h>

#include <Eigen/Core>

//FIXME add mechanisms for when the function's output is resized
//FIXME Consider the possibility of having variables in task dynamics?

namespace tvm
{

namespace task_dynamics
{

namespace abstract
{
  /** This is a base class to describe how a task is to be regulated, i.e. how
    * to compute e^(d)* for a task with constraint part f op rhs, where f is a
    * function, op is one operator among (==, <=, >=), rhs is a constant or a
    * vector and e = f-rhs. d is the order of the task dynamics.
    *
    * TaskDynamics is a lightweight descriptor, independent of a particular
    * task, that is meant for the end user.
    * Internally, it is turned into a TaskDynamicsImpl when linked to a given
    * function and rhs.
    */
  class TVM_DLLAPI TaskDynamics
  {
  public:
    virtual ~TaskDynamics() = default;

    std::unique_ptr<TaskDynamicsImpl> impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const;

    Order order() const;

  protected:
    virtual std::unique_ptr<TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const = 0;
    virtual Order order_() const = 0;
  };

}  // namespace abstract

}  // namespace task_dynamics

}  // namespace tvm

/** This macro can be used to define the derived factory required in
 * TaskDynamics implementation, Args are the arguments required by the derived
 * class, the macro arguments are members of the class passed to the derived
 * constructor */
#define TASK_DYNAMICS_DERIVED_FACTORY(...)                                                          \
  template<typename Derived, typename ... Args>                                                     \
  std::unique_ptr<tvm::task_dynamics::abstract::TaskDynamicsImpl> impl_(tvm::FunctionPtr f,         \
                                                                        tvm::constraint::Type t,    \
                                                                        const Eigen::VectorXd& rhs, \
                                                                        Args&& ... args) const      \
  {                                                                                                 \
    return std::make_unique<Derived>(f, t, rhs, std::forward<Args>(args)..., ## __VA_ARGS__);       \
  }
