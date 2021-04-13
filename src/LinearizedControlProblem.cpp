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

void LinearizedControlProblem::remove(const TaskWithRequirements & tr)
{
  ControlProblem::remove(tr);
  auto it = constraints_.find(&tr);
  if(it == constraints_.end())
  {
    return;
  }
  removeSubstitutionFor(*it->second.constraint);
  updater_.removeInput(it->second.constraint.get());
  constraints_.erase(it);
}

void LinearizedControlProblem::add(const hint::Substitution & s)
{
  substitutions_.add(s);
  notify({scheme::internal::ProblemDefinitionEvent::Type::SubstitutionAddition, &s});
  needFinalize();
}

void LinearizedControlProblem::remove(const hint::Substitution & s)
{
  substitutions_.remove(s);
  notify({scheme::internal::ProblemDefinitionEvent::Type::SubstitutionRemoval, &s});
  needFinalize();
}

const hint::internal::Substitutions & LinearizedControlProblem::substitutions() const { return substitutions_; }

void LinearizedControlProblem::removeSubstitutionFor(const constraint::abstract::LinearConstraint & cstr)
{
  for(const auto & s : substitutions_.substitutions())
  {
    const auto & cs = s.constraints();
    auto it = std::find_if(cs.begin(), cs.end(), [&](const auto & c) { return c.get() == &cstr; });
    if(it != cs.end())
    {
      if(s.isSimple())
      {
        remove(s);
      }
      else
      {
        throw std::runtime_error(
            "[LinearizedControlProblem::removeSubstitutionFor] This constraint is part of a non-simple substitution. "
            "Please remove the substitution or rearrange the substitution by hand as it is not possible to decide "
            "which variables should still be substituted after removing the constraint.");
      }
      break;
    }
  }
}

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

LinearConstraintPtr LinearizedControlProblem::constraint(const TaskWithRequirements & t) const
{
  return constraints_.at(&t).constraint;
}

LinearConstraintPtr LinearizedControlProblem::constraintNoThrow(const TaskWithRequirements & t) const
{
  auto it = constraints_.find(&t);
  if(it != constraints_.end())
    return it->second.constraint;
  else
    return {};
}

const LinearConstraintWithRequirements & LinearizedControlProblem::constraintWithRequirements(
    const TaskWithRequirements & t) const
{
  return constraints_.at(&t);
}

std::optional<std::reference_wrapper<const LinearConstraintWithRequirements>>
    LinearizedControlProblem::constraintWithRequirementsNoThrow(const TaskWithRequirements & t) const
{
  auto it = constraints_.find(&t);
  if(it != constraints_.end())
    return it->second;
  else
    return {};
}

const tvm::utils::internal::map<TaskWithRequirements const *, LinearConstraintWithRequirements> &
    LinearizedControlProblem::constraintMap() const
{
  return constraints_;
}

void LinearizedControlProblem::update_() { substitutions_.updateSubstitutions(); }

void LinearizedControlProblem::finalize_() { substitutions_.finalize(); }
} // namespace tvm
