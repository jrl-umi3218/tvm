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
    ptr->typeInfo_ = typeid(*this).hash_code();
    return ptr;
  }

}  // namespace abstract

}  // namespace task_dynamics

}  // namespace tvm
