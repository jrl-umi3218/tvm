/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/LinearizedControlProblem.h>
#include <tvm/scheme/internal/ProblemComputationData.h>
#include <tvm/utils/internal/map.h>

namespace tvm::scheme::internal
{
/** An extension of ProblemComputationData containing data specific to
 * linearized problems, in particular a mapping from task to constraint.
 *
 * \internal Keeping a local mapping is necessary, because addition and
 * removal of tasks on the problem may change the association between a
 * task and a constraint (LinearizedControlProblem recreates a constraint
 * if a task is removed and added again).
 */
class LinearizedProblemComputationData : public ProblemComputationData
{
public:
  /** Add a constraint \c c and the task \c tr it is derived from so as to keep
   * the mapping \c tr -> \c c
   */
  void addConstraint(const TaskWithRequirements & tr, const LinearConstraintWithRequirements & c)
  {
    assert(task2Constraint_.find(&tr) == task2Constraint_.end());
    task2Constraint_[&tr] = c;
  }

  /** Remove constraint corresponding to task \p tr.*/
  void removeConstraint(const TaskWithRequirements & tr) { task2Constraint_.erase(&tr); }

  /** Add a mapping task -> constraint.*/
  void addConstraints(
      const tvm::utils::internal::map<TaskWithRequirements const *, LinearConstraintWithRequirements> & map)
  {
    task2Constraint_.insert(map.begin(), map.end());
  }

  /** Access the constraint corresponding to \p tr.*/
  const LinearConstraintWithRequirements & constraint(const TaskWithRequirements & tr) const
  {
    return task2Constraint_.at(&tr);
  }

  /** Access the constraint corresponding to \p tr, and return it as a std::optional.*/
  std::optional<std::reference_wrapper<const LinearConstraintWithRequirements>> constraintNoThrow(
      const TaskWithRequirements & tr) const
  {
    auto it = task2Constraint_.find(&tr);
    if(it != task2Constraint_.end())
      return it->second;
    else
      return {};
  }

protected:
  /** Constructor, using the id of the solver.*/
  LinearizedProblemComputationData(int solverId) : ProblemComputationData(solverId) {}

  void reset()
  {
    task2Constraint_.clear();
    clearEvents();
    clearVariables();
  }

private:
  tvm::utils::internal::map<TaskWithRequirements const *, LinearConstraintWithRequirements> task2Constraint_;
};
} // namespace tvm::scheme::internal
