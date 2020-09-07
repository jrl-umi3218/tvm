/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/ControlProblem.h>
#include <tvm/defs.h>
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
  void add(TaskWithRequirementsPtr tr);
  void remove(TaskWithRequirements * tr);

  void add(const hint::Substitution & s);
  const hint::internal::Substitutions & substitutions() const;

  /** Access to the variables
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
  LinearConstraintPtr constraint(TaskWithRequirements * t) const;

  /** Access to the linear constraint corresponding to the task \p t
   *
   * \param t TaskWithRequirements object as return by add.
   *
   * \return A shared_ptr that can be null if \p t is not in the problem.
   */
  LinearConstraintPtr constraintNoThrow(TaskWithRequirements * t) const;

  /** Access to the linear constraint and requirements corresponding to the task \p t
   *
   * \param t TaskWithRequirements object as return by add.
   */
  const LinearConstraintWithRequirements & constraintWithRequirements(TaskWithRequirements * t) const;

  /** Access to the linear constraint and requirements corresponding to the task \p t
   *
   * \param t TaskWithRequirements object as return by add.
   *
   * \return An std::optional containing a const reference on LinearConstraintWithRequirements
   * if \p t is in the problem.
   */
  std::optional<std::reference_wrapper<const LinearConstraintWithRequirements>> constraintWithRequirementsNoThrow(
      TaskWithRequirements * t) const;

  /** Return the map task -> constraint*/
  const tvm::utils::internal::map<TaskWithRequirements *, LinearConstraintWithRequirements> & constraintMap() const;

protected:
  /** Compute all quantities necessary for solving the problem.*/
  void update_() override;

  /** Finalize the data of the solver*/
  void finalize_() override;

private:
  utils::internal::map<TaskWithRequirements *, LinearConstraintWithRequirements> constraints_;
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
