#include "Function.h"
#include "TaskDynamics.h"

namespace tvm
{
  TaskDynamics::TaskDynamics(TDOrder order)
    : order_(order)
  {
    registerUpdates(Update::UpdateValue, &TaskDynamics::UpdateValue);
    addOutputDependency(Output::Value, Update::UpdateValue);
  }

  void TaskDynamics::setFunction(std::shared_ptr<Function> f)
  {
    if (!f_)
    {
      f_ = f;
      addInput(f, internal::FirstOrderProvider::Output::Value); //FIXME it's not great to have to resort to internal::FirstOrderProvider
      addInputDependency(Update::UpdateValue, f, internal::FirstOrderProvider::Output::Value);
      if (order_ == TDOrder::Dynamics)
      {
        addInput(f, Function::Output::Velocity); //FIXME it's not great to have to resort to internal::FirstOrderProvider
        addInputDependency(Update::UpdateValue, f, Function::Output::Velocity);
      }
      value_.resize(f->size());
    }
    else
      throw std::runtime_error("This task dynamics was already assigned a function.");
  }

  const Eigen::VectorXd& TaskDynamics::value() const
  {
    return value_;
  }

  TDOrder TaskDynamics::order() const
  {
    return TDOrder();
  }

  Function* const TaskDynamics::function() const
  {
    return f_.get();
  }

  ProportionalDynamics::ProportionalDynamics(double kp)
    : TaskDynamics(TDOrder::Kinematics)
    , kp_(kp)
  {
  }

  void ProportionalDynamics::UpdateValue()
  {
    value_ = -kp_ * function()->value();
  }

  ProportionalDerivativeDynamics::ProportionalDerivativeDynamics(double kp, double kv)
    : TaskDynamics(TDOrder::Dynamics)
    , kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivativeDynamics::ProportionalDerivativeDynamics(double kp)
    : ProportionalDerivativeDynamics(kp, std::sqrt(2*kp))
  {
  }

  void ProportionalDerivativeDynamics::UpdateValue()
  {
    value_ = -kv_ * function()->velocity() - kp_ * function()->value();
  }
}