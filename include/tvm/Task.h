/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>

#include <tvm/constraint/enums.h>
#include <tvm/constraint/internal/RHSVectors.h>
#include <tvm/task_dynamics/abstract/TaskDynamics.h>
#include <tvm/utils/ProtoTask.h>

#include <memory>

namespace tvm
{
/** A task is a triplet (Function, operator, TaskDynamics) where operator is
 * ==, >= or <=*/
class TVM_DLLAPI Task
{
public:
  Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics & td);
  Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics & td, double rhs);
  Task(FunctionPtr f,
       constraint::Type t,
       const task_dynamics::abstract::TaskDynamics & td,
       const Eigen::VectorXd & rhs);
  Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics & td, double l, double u);
  Task(FunctionPtr f,
       constraint::Type t,
       const task_dynamics::abstract::TaskDynamics & td,
       const Eigen::VectorXd & l,
       const Eigen::VectorXd & u);
  Task(utils::ProtoTaskEQ proto, const task_dynamics::abstract::TaskDynamics & td);
  Task(utils::ProtoTaskLT proto, const task_dynamics::abstract::TaskDynamics & td);
  Task(utils::ProtoTaskGT proto, const task_dynamics::abstract::TaskDynamics & td);
  Task(utils::ProtoTaskDS proto, const task_dynamics::abstract::TaskDynamics & td);

  FunctionPtr function() const;
  constraint::Type type() const;
  TaskDynamicsPtr taskDynamics() const;
  TaskDynamicsPtr secondBoundTaskDynamics()
      const; // the dynamics of the upper bound, in the case of double-sided task only.

  template<typename T, typename TDImpl = typename T::Impl>
  std::shared_ptr<TDImpl> taskDynamics() const;

  template<typename T, typename TDImpl = typename T::Impl>
  std::shared_ptr<TDImpl> secondBoundTaskDynamics() const;

private:
  FunctionPtr f_;
  constraint::Type type_;
  TaskDynamicsPtr td_;
  TaskDynamicsPtr td2_ = nullptr; // used only for double sided tasks, as dynamics for upper bound.
};

template<typename T, typename TDImpl>
std::shared_ptr<TDImpl> Task::taskDynamics() const
{
  if(td_->checkType<TDImpl>())
    return std::static_pointer_cast<TDImpl>(td_);
  else
    throw std::runtime_error("Unable to cast the task dynamics into the desired type.");
}

template<typename T, typename TDImpl>
std::shared_ptr<TDImpl> Task::secondBoundTaskDynamics() const
{
  if(td2_->checkType<TDImpl>())
    return std::static_pointer_cast<TDImpl>(td2_);
  else
    throw std::runtime_error("Unable to cast the task dynamics into the desired type.");
}

} // namespace tvm
