#include <tvm/task_dynamics/abstract/TaskDynamics.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{

namespace abstract
{

  TaskDynamicsImpl::TaskDynamicsImpl(Order order, FunctionPtr f)
    : order_(order)
  {
    setFunction(f);
    registerUpdates(Update::UpdateValue, &TaskDynamicsImpl::updateValue);
    addOutputDependency(Output::Value, Update::UpdateValue);
  }

  void TaskDynamicsImpl::setFunction(FunctionPtr f)
  {
    if (f)
    {
      f_ = f;
      addInput(f, internal::FirstOrderProvider::Output::Value); //FIXME it's not great to have to resort to internal::FirstOrderProvider
      addInputDependency(Update::UpdateValue, f, internal::FirstOrderProvider::Output::Value);
      if (order_ == Order::Two)
      {
        addInput(f, function::abstract::Function::Output::Velocity);
        addInputDependency(Update::UpdateValue, f, function::abstract::Function::Output::Velocity);
      }
      value_.resize(f->size());
    }
    else
      throw std::runtime_error("You cannot pass a nullptr as a function.");
  }

  std::unique_ptr<TaskDynamicsImpl> TaskDynamics::impl(FunctionPtr f) const
  {
    auto ptr = impl_(f);
    ptr->typeInfo_ = typeid(*this).hash_code();
    return ptr;
  }

}  // namespace abstract

}  // namespace task_dynamics

}  // namespace tvm
