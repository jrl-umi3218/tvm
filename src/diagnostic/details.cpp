/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/diagnostic/GraphProbe.h>
#include <tvm/diagnostic/internal/details.h>

#include <tvm/constraint/BasicLinearConstraint.h>
#include <tvm/constraint/internal/LinearizedTaskConstraint.h>
#include <tvm/function/BasicLinearFunction.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/task_dynamics/Constant.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/Proportional.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/Reference.h>
#include <tvm/task_dynamics/VelocityDamper.h>

namespace tvm::diagnostic::internal
{
void registerDefault(GraphProbe & gp)
{
  gp.registerTVMConstraint<constraint::abstract::Constraint>();
  gp.registerTVMConstraint<constraint::abstract::LinearConstraint>();
  gp.registerTVMConstraint<constraint::internal::LinearizedTaskConstraint>();
  gp.registerTVMConstraint<constraint::BasicLinearConstraint>();

  gp.registerTVMFunction<function::abstract::Function>();
  gp.registerTVMFunction<function::abstract::LinearFunction>();
  gp.registerAccessor<function::abstract::LinearFunction>(function::abstract::LinearFunction::Output::B,
                                                          &function::abstract::LinearFunction::b);
  gp.registerTVMFunction<function::BasicLinearFunction>();
  gp.registerTVMFunction<function::IdentityFunction>();

  gp.registerTVMTaskDynamics<task_dynamics::Constant>();
  gp.registerTVMTaskDynamics<task_dynamics::None>();
  gp.registerTVMTaskDynamics<task_dynamics::Proportional>();
  gp.registerTVMTaskDynamics<task_dynamics::ProportionalDerivative>();
  gp.registerTVMTaskDynamics<task_dynamics::Reference>();
  gp.registerTVMTaskDynamics<task_dynamics::VelocityDamper>();
  gp.registerAccessor<task_dynamics::abstract::TaskDynamicsImpl>(
      task_dynamics::abstract::TaskDynamicsImpl::Output::Value, &task_dynamics::abstract::TaskDynamicsImpl::value);
}

} // namespace tvm::diagnostic::internal
