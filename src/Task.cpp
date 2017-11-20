#include <tvm/Task.h>

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

#include <stdexcept>

namespace
{
  Eigen::VectorXd rhs2vector(const tvm::utils::internal::RHS& rhs, tvm::FunctionPtr f)
  {
    switch (rhs.type_)
    {
    case tvm::utils::internal::RHSType::Zero: return Eigen::VectorXd::Zero(f->size());
    case tvm::utils::internal::RHSType::Double: return Eigen::VectorXd::Constant(f->size(), rhs.d_);
    case tvm::utils::internal::RHSType::Vector: return rhs.v_;
    }
  }
}

namespace tvm
{
  Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td)
    : f_(f)
    , type_(t)
    , td_(td.impl(f, t, Eigen::VectorXd::Zero(f->size())))
    , vectors_(t,constraint::RHS::AS_GIVEN)
  {
    if (t == constraint::Type::DOUBLE_SIDED)
      throw std::runtime_error("Double sided tasks need to have non-zero bounds.");
    vectors_.rhs(t) = Eigen::VectorXd::Zero(f->size());
  }

  Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td, double rhs)
    : Task(f, t, td, Eigen::VectorXd::Constant(f->size(),rhs))
  {
  }

  Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td, const Eigen::VectorXd & rhs)
    : f_(f)
    , type_(t)
    , td_(std::move(td.impl(f, t, rhs)))
    , vectors_(t, constraint::RHS::AS_GIVEN)
  {
    if (t == constraint::Type::DOUBLE_SIDED)
      throw std::runtime_error("Double sided tasks need to have two bounds.");
    vectors_.rhs(t) = rhs;
  }

  Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td, double l, double u)
    : Task(f, t, td, Eigen::VectorXd::Constant(f->size(), l), Eigen::VectorXd::Constant(f->size(), u))
  {
  }

  Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td, const Eigen::VectorXd & l, const Eigen::VectorXd & u)
    : f_(f)
    , type_(t)
    , td_(td.impl(f, constraint::Type::GREATER_THAN, l))
    , td2_(td.impl(f, constraint::Type::LOWER_THAN, l))
    , vectors_(t, constraint::RHS::AS_GIVEN)
  {
    if (t != constraint::Type::DOUBLE_SIDED)
      throw std::runtime_error("This constructor is for double sided constraints only.");
    vectors_.l() = l;
    vectors_.u() = u;
  }

  Task::Task(utils::ProtoTaskEQ proto, const task_dynamics::abstract::TaskDynamics& td)
    : Task(proto.f_, constraint::Type::EQUAL, td, rhs2vector(proto.rhs_, proto.f_))
  {
  }

  Task::Task(utils::ProtoTaskLT proto, const task_dynamics::abstract::TaskDynamics& td)
    : Task(proto.f_, constraint::Type::LOWER_THAN, td, rhs2vector(proto.rhs_, proto.f_))
  {
  }

  Task::Task(utils::ProtoTaskGT proto, const task_dynamics::abstract::TaskDynamics& td)
    : Task(proto.f_, constraint::Type::GREATER_THAN, td, rhs2vector(proto.rhs_, proto.f_))
  {
  }

  Task::Task(utils::ProtoTaskDS proto, const task_dynamics::abstract::TaskDynamics& td)
    : Task(proto.f_, constraint::Type::DOUBLE_SIDED, td, 
           rhs2vector(proto.l_, proto.f_), rhs2vector(proto.u_, proto.f_))
  {
  }


  FunctionPtr Task::function() const
  {
    return f_;
  }

  constraint::Type Task::type() const
  {
    return type_;
  }

  TaskDynamicsPtr Task::taskDynamics() const
  {
    return td_;
  }

}  // namespace tvm
