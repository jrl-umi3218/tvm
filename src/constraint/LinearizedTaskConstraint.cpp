#include <tvm/constraint/internal/LinearizedTaskConstraint.h>

#include <tvm/Variable.h>
#include <tvm/function/abstract/Function.h>
#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm
{

namespace constraint
{

namespace internal
{

  LinearizedTaskConstraint::LinearizedTaskConstraint(const Task& task)
    : LinearConstraint(task.type(), constraint::RHS::AS_GIVEN, task.function()->size())
    , f_(task.function())
    , td_(task.taskDynamics())
  {
    typedef LinearizedTaskConstraint LTC;
    Constraint::Output output;
    void (LTC::*kin)();
    void (LTC::*dyn)();

    switch (task.type())
    {
    case constraint::Type::GREATER_THAN:
      output = Constraint::Output::L;
      kin = &LTC::updateLKin;
      dyn = &LTC::updateLDyn;
      break;
    case constraint::Type::LOWER_THAN:
      output = Constraint::Output::U;
      kin = &LTC::updateUKin;
      dyn = &LTC::updateUDyn;
      break;
    case constraint::Type::EQUAL:
      kin = &LTC::updateEKin;
      dyn = &LTC::updateEDyn;
      output = Constraint::Output::E;
      break;
    default: assert(false); break;
    }

    switch (task.taskDynamics()->order())
    {
    case task_dynamics::Order::Geometric:
      throw std::runtime_error("This case is not implemented yet.");
    case task_dynamics::Order::Kinematics:
    {
      for (auto& v : f_->variables())
        addVariable(dot(v), true);
      registerUpdates(Update::UpdateRHS, kin);
    }
    break;
    case task_dynamics::Order::Dynamics:
    {
      for (auto& v : f_->variables())
        addVariable(dot(v, 2), true);
      registerUpdates(Update::UpdateRHS, dyn);
      addInputDependency<LTC>(Update::UpdateRHS, f_, function::abstract::Function::Output::NormalAcceleration);
    }
    break;
    }

    using BaseOutput = tvm::internal::FirstOrderProvider::Output;
    addInputDependency<LTC>(Update::UpdateRHS, td_, task_dynamics::abstract::TaskDynamics::Output::Value);
    addInputDependency<LinearConstraint>(LinearConstraint::Update::Value, f_, BaseOutput::Jacobian);
#ifndef WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wpragmas"
# pragma GCC diagnostic ignored "-Wunknown-warning-option"
# pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    addOutputDependency<LTC>(output, Update::UpdateRHS);
#ifndef WIN32
# pragma GCC diagnostic pop
#endif
    addDirectDependency<LTC>(BaseOutput::Jacobian, f_, BaseOutput::Jacobian);
  }

  LinearizedTaskConstraint::LinearizedTaskConstraint(const ProtoTask& pt, TaskDynamicsPtr td)
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

  const tvm::internal::MatrixWithProperties& LinearizedTaskConstraint::jacobian(const Variable& x) const
  {
    switch (td_->order())
    {
    case task_dynamics::Order::Geometric: return f_->jacobian(x); break;
    case task_dynamics::Order::Kinematics: return f_->jacobian(*x.primitive()); break;
    case task_dynamics::Order::Dynamics: return f_->jacobian(*x.primitive<2>()); break;
    default:
      throw std::runtime_error("Unimplemented case.");
    }
  }

}  // namespace internal

}  // namespace constraint

}  // namespace tvm
