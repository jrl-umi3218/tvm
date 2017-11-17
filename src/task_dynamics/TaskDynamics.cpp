#include <tvm/task_dynamics/abstract/TaskDynamics.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{

namespace abstract
{

  TaskDynamics::TaskDynamics(Order order)
    : order_(order)
  {
    registerUpdates(Update::UpdateValue, &TaskDynamics::updateValue);
    addOutputDependency(Output::Value, Update::UpdateValue);
  }

  void TaskDynamics::setFunction(FunctionPtr f)
  {
    if (!f_)
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
      throw std::runtime_error("This task dynamics was already assigned a function.");

    setFunction_();
  }

}  // namespace abstract

}  // namespace task_dynamics

}  // namespace tvm
