#include <tvm/Task.h>

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

#include <stdexcept>

namespace tvm
{
  ProtoTask operator==(FunctionPtr f, double rhs)
  {
    if (rhs == 0)
      return{ f, constraint::Type::EQUAL };
    else
      throw std::runtime_error("Only 0 is supported as a right hand side.");
  }

  ProtoTask operator>=(FunctionPtr f, double rhs)
  {
    if (rhs == 0)
      return{ f, constraint::Type::GREATER_THAN };
    else
      throw std::runtime_error("Only 0 is supported as a right hand side.");
  }

  ProtoTask operator<=(FunctionPtr f, double rhs)
  {
    if (rhs == 0)
      return{ f, constraint::Type::LOWER_THAN };
    else
      throw std::runtime_error("Only 0 is supported as a right hand side.");
  }

  Task::Task(FunctionPtr f, constraint::Type t, TaskDynamicsPtr td)
    : f_(f)
    , type_(t)
    , td_(td)
    , vectors_(t,constraint::RHS::ZERO)
  {
    if (t == constraint::Type::DOUBLE_SIDED)
      throw std::runtime_error("Double sided tasks need to have non-zero bounds.");
    td->setFunction(f);
  }

  Task::Task(ProtoTask proto, TaskDynamicsPtr td)
    :Task(proto.f_, proto.type_, td)
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
