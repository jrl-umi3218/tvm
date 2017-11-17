#include <tvm/task_dynamics/abstract/TaskDynamics.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{

namespace abstract
{

  std::unique_ptr<TaskDynamicsImpl> TaskDynamics::impl(FunctionPtr f) const
  {
    auto ptr = impl_(f);
    ptr->typeInfo_ = typeid(*this).hash_code();
    return ptr;
  }

}  // namespace abstract

}  // namespace task_dynamics

}  // namespace tvm
