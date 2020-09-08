/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Task.h>

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

#include <stdexcept>

namespace tvm
{
Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics & td)
: f_(f), type_(t), td_(td.impl(f, t, Eigen::VectorXd::Zero(f->size())))
{
  if(t == constraint::Type::DOUBLE_SIDED)
    throw std::runtime_error("Double sided tasks need to have non-zero bounds.");
  if(!f->imageSpace().isEuclidean())
    throw std::runtime_error("[Task::Task] Can't create a task for a function into a non-Euclidean space.");
}

Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics & td, double rhs)
: Task(f, t, td, Eigen::VectorXd::Constant(f->size(), rhs))
{}

Task::Task(FunctionPtr f,
           constraint::Type t,
           const task_dynamics::abstract::TaskDynamics & td,
           const Eigen::VectorXd & rhs)
: f_(f), type_(t), td_(std::move(td.impl(f, t, rhs)))
{
  if(t == constraint::Type::DOUBLE_SIDED)
    throw std::runtime_error("Double sided tasks need to have two bounds.");
  if(!f->imageSpace().isEuclidean())
    throw std::runtime_error("[Task::Task] Can't create a task for a function into a non-Euclidean space.");
}

Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics & td, double l, double u)
: Task(f, t, td, Eigen::VectorXd::Constant(f->size(), l), Eigen::VectorXd::Constant(f->size(), u))
{}

Task::Task(FunctionPtr f,
           constraint::Type t,
           const task_dynamics::abstract::TaskDynamics & td,
           const Eigen::VectorXd & l,
           const Eigen::VectorXd & u)
: f_(f), type_(t), td_(td.impl(f, constraint::Type::GREATER_THAN, l)), td2_(td.impl(f, constraint::Type::LOWER_THAN, u))
{
  if(t != constraint::Type::DOUBLE_SIDED)
    throw std::runtime_error("This constructor is for double sided constraints only.");
  if(!f->imageSpace().isEuclidean())
    throw std::runtime_error("[Task::Task] Can't create a task for a function into a non-Euclidean space.");
}

Task::Task(utils::ProtoTaskEQ proto, const task_dynamics::abstract::TaskDynamics & td)
: Task(proto.f_, constraint::Type::EQUAL, td, proto.rhs_.toVector(proto.f_->size()))
{}

Task::Task(utils::ProtoTaskLT proto, const task_dynamics::abstract::TaskDynamics & td)
: Task(proto.f_, constraint::Type::LOWER_THAN, td, proto.rhs_.toVector(proto.f_->size()))
{}

Task::Task(utils::ProtoTaskGT proto, const task_dynamics::abstract::TaskDynamics & td)
: Task(proto.f_, constraint::Type::GREATER_THAN, td, proto.rhs_.toVector(proto.f_->size()))
{}

Task::Task(utils::ProtoTaskDS proto, const task_dynamics::abstract::TaskDynamics & td)
: Task(proto.f_,
       constraint::Type::DOUBLE_SIDED,
       td,
       proto.l_.toVector(proto.f_->size()),
       proto.u_.toVector(proto.f_->size()))
{}

FunctionPtr Task::function() const { return f_; }

constraint::Type Task::type() const { return type_; }

TaskDynamicsPtr Task::taskDynamics() const { return td_; }

TaskDynamicsPtr Task::secondBoundTaskDynamics() const { return td2_; }

} // namespace tvm
