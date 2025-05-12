/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/defs.h>

#include <tvm/ControlProblem.h>
#include <tvm/hint/Substitution.h>
#include <tvm/hint/internal/Substitutions.h>

#include <optional>

namespace tvm
{
class TVM_DLLAPI LinearConstraintWithRequirements
{
public:
  LinearConstraintPtr constraint;
  SolvingRequirementsPtr requirements;
  bool bound;
};

class TVM_DLLAPI LinearizedControlProblem : public ControlProblem
{
public:
  LinearizedControlProblem();
  LinearizedControlProblem(const ControlProblem & pb);

  TaskWithRequirementsPtr add(const Task & task, const requirements::SolvingRequirements & req = {});
  template<constraint::Type T>
  TaskWithRequirementsPtr add(utils::ProtoTask<T> proto,
                              const task_dynamics::abstract::TaskDynamics & td,
                              const requirements::SolvingRequirements & req = {});
  template<constraint::Type T>
  TaskWithRequirementsPtr add(utils::LinearProtoTask<T> proto,
                              const task_dynamics::abstract::TaskDynamics & td,
                              const requirements::SolvingRequirements & req = {});
  template<constraint::Type T>
  TaskWithRequirementsPtr add(utils::LinearProtoTask<T> proto, const requirements::SolvingRequirements & req = {});
  void add(TaskWithRequirementsPtr tr, bool notify = true);
  void remove(const TaskWithRequirements & tr, bool notify = true);

  void add(const hint::Substitution & s);
  void remove(const hint::Substitution & s);
  /**
   * If the task's function has changed (added variables, etc), then we need to recreate
   * the associated constraint
   * */
  void updateConstraint(const TaskWithRequirements & task);

  const hint::internal::Substitutions & substitutions() const;
  void removeSubstitutionFor(const constraint::abstract::LinearConstraint & cstr);


  /** Access to the variables of the problem.
   *
   * \note These are all the variables irrespective of any substitutions as
   * substitutions are hints for the solver.
   *
   * \note The result is not cached, i.e. it is recomputed at each call.
   */
  VariableVector variables() const;

  /** Access to constraints*/
  std::vector<LinearConstraintWithRequirements> constraints() const;

  /** Access to the linear constraint corresponding to the task \p t
   *
   * \param t TaskWithRequirements object as return by add.
   */
  LinearConstraintPtr constraint(const TaskWithRequirements & t) const;

  /** Access to the linear constraint corresponding to the task \p t
   *
   * \param t TaskWithRequirements object as return by add.
   *
   * \return A shared_ptr that can be null if \p t is not in the problem.
   */
  LinearConstraintPtr constraintNoThrow(const TaskWithRequirements & t) const;

  /** Access to the linear constraint and requirements corresponding to the task \p t
   *
   * \param t TaskWithRequirements object as return by add.
   */
  const LinearConstraintWithRequirements & constraintWithRequirements(const TaskWithRequirements & t) const;

  /** Access to the linear constraint and requirements corresponding to the task \p t
   *
   * \param t TaskWithRequirements object as return by add.
   *
   * \return An std::optional containing a const reference on LinearConstraintWithRequirements
   * if \p t is in the problem.
   */
  std::optional<std::reference_wrapper<const LinearConstraintWithRequirements>> constraintWithRequirementsNoThrow(
      const TaskWithRequirements & t) const;

  /** Return the map task -> constraint*/
  const tvm::utils::internal::map<TaskWithRequirements const *, LinearConstraintWithRequirements> & constraintMap()
      const;

protected:
  /** Compute all quantities necessary for solving the problem.*/
  void update_() override;

  /** Finalize the data of the solver*/
  void finalize_() override;

private:
  utils::internal::map<TaskWithRequirements const *, LinearConstraintWithRequirements> constraints_;
  hint::internal::Substitutions substitutions_;
};

template<constraint::Type T>
TaskWithRequirementsPtr LinearizedControlProblem::add(utils::ProtoTask<T> proto,
                                                      const task_dynamics::abstract::TaskDynamics & td,
                                                      const requirements::SolvingRequirements & req)
{
  return add({proto, td}, req);
}

template<constraint::Type T>
TaskWithRequirementsPtr LinearizedControlProblem::add(utils::LinearProtoTask<T> proto,
                                                      const task_dynamics::abstract::TaskDynamics & td,
                                                      const requirements::SolvingRequirements & req)
{
  return add({proto, td}, req);
}

template<constraint::Type T>
TaskWithRequirementsPtr LinearizedControlProblem::add(utils::LinearProtoTask<T> proto,
                                                      const requirements::SolvingRequirements & req)
{
  return add({proto, task_dynamics::None()}, req);
}
} // namespace tvm
