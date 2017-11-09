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

    switch (task.taskDynamics()->order())
    {
    case TDOrder::Geometric:
      throw std::runtime_error("This case is not implemented yet.");
    case TDOrder::Kinematics:
    {
      for (auto& v : f_->variables())
        addVariable(dot(v), true);
      registerUpdates(Update::UpdateRHS, kin);
    }
    break;
    case TDOrder::Dynamics:
    {
      for (auto& v : f_->variables())
        addVariable(dot(v, 2), true);
      registerUpdates(Update::UpdateRHS, dyn);
      addInputDependency<LTC>(Update::UpdateRHS, f_, Function::Output::NormalAcceleration);
    }
    break;
    }

    using BaseOutput = internal::FirstOrderProvider::Output;
    addInputDependency<LTC>(Update::UpdateRHS, td_, TaskDynamics::Output::Value);
    addInputDependency<LinearConstraint>(LinearConstraint::Update::Value, f_, BaseOutput::Jacobian);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    addOutputDependency<LTC>(output, Update::UpdateRHS);
#pragma GCC diagnostic pop
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

  const MatrixWithProperties& LinearizedTaskConstraint::jacobian(const Variable& x) const
  {
    switch (td_->order())
    {
    case TDOrder::Geometric: return f_->jacobian(x); break;
    case TDOrder::Kinematics: return f_->jacobian(*x.primitive()); break;
    case TDOrder::Dynamics: return f_->jacobian(*x.primitive<2>()); break;
    default:
      throw std::runtime_error("Unimplemented case.");
    }
  }
}
