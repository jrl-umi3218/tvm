/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

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
    if (type() == constraint::Type::DOUBLE_SIDED)
    {
      td2_ = task.secondBoundTaskDynamics();
      if (td_->order() != td2_->order())
      {
        throw std::runtime_error("For double-sided task, the dynamic of both sides must have the same order.");
      }
    }

    using LTC = LinearizedTaskConstraint;
    Constraint::Output_ output;
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
    case constraint::Type::DOUBLE_SIDED:
      kin = &LTC::updateLKin;
      dyn = &LTC::updateLDyn;
      output = Constraint::Output::L;
      break;
    default:
      assert(false);
      break;
    }

    switch (td_->order())
    {
    case task_dynamics::Order::Zero:
    {
      for (auto& v : f_->variables())
      {
        if (!f_->linearIn(*v))
          throw std::runtime_error("The function is not linear in " + v->name());
        addVariable(v, true);
      }
      registerUpdates(Update::UpdateRHS, kin);
    }
    break;
    case task_dynamics::Order::One:
    {
      for (auto& v : f_->variables())
        addVariable(dot(v), true);
      registerUpdates(Update::UpdateRHS, kin);
    }
    break;
    case task_dynamics::Order::Two:
    {
      for (auto& v : f_->variables())
        addVariable(dot(v, 2), true);
      registerUpdates(Update::UpdateRHS, dyn);
      addInputDependency<LTC>(Update::UpdateRHS, f_, function::abstract::Function::Output::NormalAcceleration);
    }
    break;
    }

    using BaseOutput = tvm::internal::FirstOrderProvider::Output;
    addInputDependency<LTC>(Update::UpdateRHS, td_, task_dynamics::abstract::TaskDynamicsImpl::Output::Value);
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

    if (type() == constraint::Type::DOUBLE_SIDED)
    {
      if (td2_->order() == task_dynamics::Order::Two)
      {
        registerUpdates(Update::UpdateRHS2, &LTC::updateU2Dyn);
        addInputDependency<LTC>(Update::UpdateRHS2, f_, function::abstract::Function::Output::NormalAcceleration);
      }
      else
      {
        registerUpdates(Update::UpdateRHS2, &LTC::updateU2Kin);
      }
      addInputDependency<LTC>(Update::UpdateRHS2, td2_, task_dynamics::abstract::TaskDynamicsImpl::Output::Value);
#ifndef WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wpragmas"
# pragma GCC diagnostic ignored "-Wunknown-warning-option"
# pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
      addOutputDependency<LTC>(Constraint::Output::U, Update::UpdateRHS2);
#ifndef WIN32
# pragma GCC diagnostic pop
#endif
    }
  }

  void LinearizedTaskConstraint::updateLKin()
  {
    lRef() = td_->value();
  }

  void LinearizedTaskConstraint::updateLDyn()
  {
    lRef() = td_->value() - f_->normalAcceleration();
  }

  void LinearizedTaskConstraint::updateUKin()
  {
    uRef() = td_->value();
  }

  void LinearizedTaskConstraint::updateUDyn()
  {
    uRef() = td_->value() - f_->normalAcceleration();
  }

  void LinearizedTaskConstraint::updateEKin()
  {
    eRef() = td_->value();
  }

  void LinearizedTaskConstraint::updateEDyn()
  {
    eRef() = td_->value() - f_->normalAcceleration();
  }

  void LinearizedTaskConstraint::updateU2Kin()
  {
    uRef() = td2_->value();
  }

  void LinearizedTaskConstraint::updateU2Dyn()
  {
    uRef() = td2_->value() - f_->normalAcceleration();
  }

  const tvm::internal::MatrixWithProperties& LinearizedTaskConstraint::jacobian(const Variable& x) const
  {
    switch (td_->order())
    {
    case task_dynamics::Order::Zero: return f_->jacobian(x); break;
    case task_dynamics::Order::One: return f_->jacobian(*x.primitive()); break;
    case task_dynamics::Order::Two: return f_->jacobian(*x.primitive<2>()); break;
    default:
      throw std::runtime_error("Unimplemented case.");
    }
  }

}  // namespace internal

}  // namespace constraint

}  // namespace tvm
