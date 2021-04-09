/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/LinearizedControlProblem.h>

#include <tvm/constraint/internal/LinearizedTaskConstraint.h>
#include <tvm/scheme/internal/LinearizedProblemComputationData.h>
#include <tvm/scheme/internal/helpers.h>

namespace tvm
{
LinearizedControlProblem::LinearizedControlProblem() {}

LinearizedControlProblem::LinearizedControlProblem(const ControlProblem & pb) : ControlProblem()
{
  for(auto tr : pb.tasks())
    add(tr);
}

TaskWithRequirementsPtr LinearizedControlProblem::add(const Task & task, const requirements::SolvingRequirements & req)
{
  auto tr = std::make_shared<TaskWithRequirements>(task, req);
  add(tr);
  return tr;
}

void LinearizedControlProblem::add(TaskWithRequirementsPtr tr)
{
  ControlProblem::add(tr);

  // FIXME A lot of work can be done here based on the properties of the task's jacobian.
  // In particular, we could detect bounds, pairs of tasks forming a double-sided constraints...
  LinearConstraintWithRequirements lcr;
  lcr.constraint = std::make_shared<constraint::internal::LinearizedTaskConstraint>(tr->task);
  // we use the aliasing constructor of std::shared_ptr to ensure that
  // lcr.requirements points to and doesn't outlive tr->requirements.
  lcr.requirements = std::shared_ptr<requirements::SolvingRequirementsWithCallbacks>(tr, &tr->requirements);
  lcr.bound = scheme::internal::isBound(lcr.constraint);
  constraints_[tr.get()] = lcr;

  using CstrOutput = internal::FirstOrderProvider::Output;
  updater_.addInput(lcr.constraint, CstrOutput::Jacobian);
  switch(tr->task.type())
  {
    case constraint::Type::EQUAL:
      updater_.addInput(lcr.constraint, constraint::abstract::Constraint::Output::E);
      break;
    case constraint::Type::GREATER_THAN:
      updater_.addInput(lcr.constraint, constraint::abstract::Constraint::Output::L);
      break;
    case constraint::Type::LOWER_THAN:
      updater_.addInput(lcr.constraint, constraint::abstract::Constraint::Output::U);
      break;
    case constraint::Type::DOUBLE_SIDED:
      updater_.addInput(lcr.constraint, constraint::abstract::Constraint::Output::L,
                        constraint::abstract::Constraint::Output::U);
      break;
  }
}

void LinearizedControlProblem::remove(TaskWithRequirements * tr)
{
  ControlProblem::remove(tr);
  auto it = constraints_.find(tr);
  if(it == constraints_.end())
  {
    return;
  }
  updater_.removeInput(it->second.constraint.get());
  constraints_.erase(it);
}

void LinearizedControlProblem::add(const hint::Substitution & s) { substitutions_.add(s); }

const hint::internal::Substitutions & LinearizedControlProblem::substitutions() const { return substitutions_; }

VariableVector LinearizedControlProblem::variables() const
{
  tvm::internal::VariableCountingVector variables;
  for(auto c : constraints_)
    variables.add(c.second.constraint->variables());

  return variables.variables();
}

std::vector<LinearConstraintWithRequirements> LinearizedControlProblem::constraints() const
{
  std::vector<LinearConstraintWithRequirements> constraints;
  constraints.reserve(constraints_.size());
  for(auto c : constraints_)
    constraints.push_back(c.second);

  return constraints;
}

LinearConstraintPtr LinearizedControlProblem::constraint(TaskWithRequirements * t) const
{
  return constraints_.at(t).constraint;
}

LinearConstraintPtr LinearizedControlProblem::constraintNoThrow(TaskWithRequirements * t) const
{
  auto it = constraints_.find(t);
  if(it != constraints_.end())
    return it->second.constraint;
  else
    return {};
}

const LinearConstraintWithRequirements & LinearizedControlProblem::constraintWithRequirements(
    TaskWithRequirements * t) const
{
  return constraints_.at(t);
}

std::optional<std::reference_wrapper<const LinearConstraintWithRequirements>>
    LinearizedControlProblem::constraintWithRequirementsNoThrow(TaskWithRequirements * t) const
{
  auto it = constraints_.find(t);
  if(it != constraints_.end())
    return it->second;
  else
    return {};
}

const tvm::utils::internal::map<TaskWithRequirements *, LinearConstraintWithRequirements> &
    LinearizedControlProblem::constraintMap() const
{
  return constraints_;
}

void LinearizedControlProblem::update_() { substitutions_.updateSubstitutions(); }

void LinearizedControlProblem::finalize_() { substitutions_.finalize(); }
} // namespace tvm
