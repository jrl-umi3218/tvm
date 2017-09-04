#include <stdexcept>

#include "Constraint.h"
#include "Task.h"
#include "TaskDynamics.h"

namespace tvm
{
  ProtoTask operator==(FunctionPtr f, double rhs)
  {
    if (rhs == 0)
      return{ f, ConstraintType::EQUAL };
    else
      throw std::runtime_error("Only 0 is supported as a right hand side.");
  }

  ProtoTask operator>=(FunctionPtr f, double rhs)
  {
    if (rhs == 0)
      return{ f, ConstraintType::GREATER_THAN };
    else
      throw std::runtime_error("Only 0 is supported as a right hand side.");
  }

  ProtoTask operator<=(FunctionPtr f, double rhs)
  {
    if (rhs == 0)
      return{ f, ConstraintType::LOWER_THAN };
    else
      throw std::runtime_error("Only 0 is supported as a right hand side.");
  }

  Task::Task(FunctionPtr f, ConstraintType t, std::shared_ptr<TaskDynamics> td)
    : f_(f)
    , type_(t)
    , td_(td)
  {
    if (t == ConstraintType::DOUBLE_SIDED)
      throw std::runtime_error("Double sided tasks are not supported for now.");
    td->setFunction(f);
  }

  Task::Task(ProtoTask proto, std::shared_ptr<TaskDynamics> td)
    :Task(proto.f_, proto.type_, td)
  {
  }

  FunctionPtr Task::function() const
  {
    return f_;
  }

  ConstraintType Task::type() const
  {
    return type_;
  }

  std::shared_ptr<TaskDynamics> Task::taskDynamics() const
  {
    return td_;
  }
}