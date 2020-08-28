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

#include <tvm/LinearizedControlProblem.h>

#include <tvm/constraint/internal/LinearizedTaskConstraint.h>
#include <tvm/scheme/internal/helpers.h>
#include <tvm/scheme/internal/LinearizedProblemComputationData.h>

namespace tvm
{
  LinearizedControlProblem::LinearizedControlProblem()
  {
  }

  LinearizedControlProblem::LinearizedControlProblem(const ControlProblem& pb)
  : ControlProblem()
  {
    for (auto tr : pb.tasks())
      add(tr);
  }

  TaskWithRequirementsPtr LinearizedControlProblem::add(const Task& task, const requirements::SolvingRequirements& req)
  {
    auto tr = std::make_shared<TaskWithRequirements>(task, req);
    add(tr);
    return tr;
  }

  void LinearizedControlProblem::add(TaskWithRequirementsPtr tr)
  {
    ControlProblem::add(tr);

    //FIXME A lot of work can be done here based on the properties of the task's jacobian.
    //In particular, we could detect bounds, pairs of tasks forming a double-sided constraints...
    LinearConstraintWithRequirements lcr;
    lcr.constraint = std::make_shared<constraint::internal::LinearizedTaskConstraint>(tr->task);
    //we use the aliasing constructor of std::shared_ptr to ensure that
    //lcr.requirements points to and doesn't outlive tr->requirements.
    lcr.requirements = std::shared_ptr<requirements::SolvingRequirementsWithCallbacks>(tr, &tr->requirements);
    lcr.bound = scheme::internal::isBound(lcr.constraint);
    constraints_[tr.get()] = lcr;

    using CstrOutput = internal::FirstOrderProvider::Output;
    updater_.addInput(lcr.constraint, CstrOutput::Jacobian);
    switch (tr->task.type())
    {
    case constraint::Type::EQUAL: updater_.addInput(lcr.constraint, constraint::abstract::Constraint::Output::E); break;
    case constraint::Type::GREATER_THAN: updater_.addInput(lcr.constraint, constraint::abstract::Constraint::Output::L); break;
    case constraint::Type::LOWER_THAN: updater_.addInput(lcr.constraint, constraint::abstract::Constraint::Output::U); break;
    case constraint::Type::DOUBLE_SIDED: updater_.addInput(lcr.constraint, constraint::abstract::Constraint::Output::L, constraint::abstract::Constraint::Output::U); break;
    }
  }

  void LinearizedControlProblem::remove(TaskWithRequirements* tr)
  {
    ControlProblem::remove(tr);
    // if the above line did not throw, tr exists in the problem and in constraints_
    auto it = constraints_.find(tr);
    assert(it != constraints_.end());
    updater_.removeInput(it->second.constraint.get());
    for (auto& c : computationData_)
      static_cast<scheme::internal::LinearizedProblemComputationData*>(c.second.get())->transferRemovedConstraint({ tr, it->second });
    constraints_.erase(it);
  }

  void LinearizedControlProblem::add(const hint::Substitution & s)
  {
    substitutions_.add(s);
  }

  const hint::internal::Substitutions & LinearizedControlProblem::substitutions() const
  {
    return substitutions_;
  }

  VariableVector LinearizedControlProblem::variables() const
  {
    VariableVector variables;
    for (auto c : constraints_)
      variables.add(c.second.constraint->variables());

    return variables;
  }

  std::vector<LinearConstraintWithRequirements> LinearizedControlProblem::constraints() const
  {
    std::vector<LinearConstraintWithRequirements> constraints;
    constraints.reserve(constraints_.size());
    for (auto c: constraints_)
      constraints.push_back(c.second);

    return constraints;
  }

  LinearConstraintPtr LinearizedControlProblem::constraint(TaskWithRequirements* t) const
  {
    return constraints_.at(t).constraint;
  }

  const LinearConstraintWithRequirements& LinearizedControlProblem::constraintWithRequirements(TaskWithRequirements* t) const
  {
    return constraints_.at(t);
  }

  void LinearizedControlProblem::update_()
  {
    substitutions_.updateSubstitutions();
  }

  void LinearizedControlProblem::finalize_()
  {
    substitutions_.finalize();
  }
}  // namespace tvm
