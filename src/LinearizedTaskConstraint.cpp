#include "Function.h"
#include "LinearizedTaskConstraint.h"
#include "TaskDynamics.h"
#include "Variable.h"


namespace tvm
{
  LinearizedTaskConstraint::LinearizedTaskConstraint(const Task& task)
    : LinearConstraint(task.type(), ConstraintRHS::AS_GIVEN, task.function()->size())
    , f_(task.function())
    , td_(task.taskDynamics())
  {
    typedef LinearizedTaskConstraint LTC;
    Constraint::Output output;
    void (LTC::*kin)();
    void (LTC::*dyn)();

    switch (task.type())
    {
    case ConstraintType::GREATER_THAN: 
      output = Constraint::Output::L; 
      kin = &LTC::updateLKin;
      dyn = &LTC::updateLDyn;
      break;
    case ConstraintType::LOWER_THAN: 
      output = Constraint::Output::U;
      kin = &LTC::updateUKin;
      dyn = &LTC::updateUDyn;
      break;
    case ConstraintType::EQUAL: 
      kin = &LTC::updateEKin;
      dyn = &LTC::updateEDyn;
      output = Constraint::Output::E; 
      break;
    default: assert(false); break;
    }

    if (task.taskDynamics()->order() == TDOrder::Kinematics)
    {
      for (auto& v : f_->variables())
        addVariable(dot(v));
      registerUpdates(Update::UpdateRHS, kin);
    }
    else
    {
      for (auto& v : f_->variables())
        addVariable(dot(v,2));
      registerUpdates(Update::UpdateRHS, dyn);
      addInputDependency<LTC>(Update::UpdateRHS, f_, Function::Output::NormalAcceleration);
    }
    using BaseOutput = internal::FirstOrderProvider::Output;
    addInputDependency<LTC>(Update::UpdateRHS, f_, BaseOutput::Value);
    addOutputDependency<LTC>(output, Update::UpdateRHS);
    addDirectDependency<LTC>(BaseOutput::Jacobian, f_, BaseOutput::Jacobian);
  }

  LinearizedTaskConstraint::LinearizedTaskConstraint(const ProtoTask& pt, std::shared_ptr<TaskDynamics> td)
    : LinearizedTaskConstraint(Task(pt, td))
  {
  }

  void LinearizedTaskConstraint::updateLKin()
  {
    l_ = td_->value();
  }

  void LinearizedTaskConstraint::updateLDyn()
  {
    l_ = td_->value() - f_->normalAcceleration();
  }
  
  void LinearizedTaskConstraint::updateUKin()
  {
    u_ = td_->value();
  }
  
  void LinearizedTaskConstraint::updateUDyn()
  {
    u_ = td_->value() - f_->normalAcceleration();
  }
  
  void LinearizedTaskConstraint::updateEKin()
  {
    e_ = td_->value();
  }
  
  void LinearizedTaskConstraint::updateEDyn()
  {
    e_ = td_->value() - f_->normalAcceleration();
  }

  const Eigen::MatrixXd& LinearizedTaskConstraint::jacobian(const Variable& x) const
  {
    return f_->jacobian(x);
  }
}