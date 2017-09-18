#include "Function.h"
#include "TaskDynamics.h"

namespace tvm
{
  TaskDynamics::TaskDynamics(TDOrder order)
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
      if (order_ == TDOrder::Dynamics)
      {
        addInput(f, Function::Output::Velocity);
        addInputDependency(Update::UpdateValue, f, Function::Output::Velocity);
      }
      value_.resize(f->size());
    }
    else
      throw std::runtime_error("This task dynamics was already assigned a function.");

    setFunction_();
  }


  NoDynamics::NoDynamics(const Eigen::VectorXd& v)
    : TaskDynamics(TDOrder::Geometric)
  {
    value_ = v;
  }

  void NoDynamics::updateValue()
  {
    // do nothing
  }

  void NoDynamics::setFunction_()
  {
    if (value_.size() == 0)
      value_.setZero(function()->size());
    else
      assert(value_.size() == function()->size());
  }

  ProportionalDynamics::ProportionalDynamics(double kp)
    : TaskDynamics(TDOrder::Kinematics)
    , kp_(kp)
  {
  }

  void ProportionalDynamics::updateValue()
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

  void ProportionalDerivativeDynamics::updateValue()
  {
    value_ = -kv_ * function()->velocity() - kp_ * function()->value();
  }
}