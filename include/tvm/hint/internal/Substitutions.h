/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>

#include <tvm/graph/internal/DependencyGraph.h>
#include <tvm/hint/Substitution.h>
#include <tvm/hint/internal/SubstitutionUnit.h>

#include <vector>

namespace tvm
{

namespace hint
{

namespace internal
{
/** A set of substitutions*/
class TVM_DLLAPI Substitutions
{
public:
  /** Add substitution \p s*/
  void add(const Substitution & s);

  /** Remove all substitutions associated to the given \p cstr */
  bool remove(LinearConstraintPtr cstr);

  /** Get the vector of all substitutions, as added.
   * Note that it is not necessarily the vector of substitutions actually
   * used, as it might be needed to group substitutions (when a group of
   * substitutions depends on each other variables).
   */
  const std::vector<Substitution> & substitutions() const;

  /** Return \p true if \p c is used in one of the substitutions*/
  bool uses(LinearConstraintPtr c) const;

  /** Compute all the data needed for the substitutions.
   * Needs to be called after all the call to \p add, and before the calls to
   * \p variables, \p variableSubstitutions and \p additionalConstraints.
   */
  void finalize();

  /** Update the data for the substitutions*/
  void updateSubstitutions();

  /** Update the value of the substituted variables according to the values of
   * the non-substituted ones.*/
  void updateVariableValues() const;

  /** All variables x in the substitutions*/
  const std::vector<VariablePtr> & variables() const;
  /** The linear functions x = f(y,z) corresponding to the variables*/
  const std::vector<std::shared_ptr<function::BasicLinearFunction>> & variableSubstitutions() const;
  /** The additional nullspace variables z*/
  const std::vector<VariablePtr> & additionalVariables() const;
  /** The remaining constraints on y and z*/
  const std::vector<std::shared_ptr<constraint::BasicLinearConstraint>> & additionalConstraints() const;
  /** The variables y*/
  const std::vector<VariablePtr> & otherVariables() const;
  /** If \p x is one of the substituted variables, returns the variables it is
   * replaced by. Otherwise, return \p x
   */
  VariableVector substitute(const VariablePtr & x) const;

private:
  /** The substitutions, as added to the objects*/
  std::vector<Substitution> substitutions_;

  /** Dependency graph between the substitutions. There is an edge from i to j
   * if the substitutions_[i] relies on substitutions_[j].
   */
  tvm::graph::internal::DependencyGraph dependencies_;

  /** Group of dependent substitutions*/
  std::vector<SubstitutionUnit> units_;

  /** The variables substituted (x).*/
  std::vector<VariablePtr> variables_;

  /** The substitution functions linked to the variables, i.e
   * variables_[i].value() is given by varSubstitutions_[i].value().
   */
  std::vector<std::shared_ptr<function::BasicLinearFunction>> varSubstitutions_;

  /** Nullspace variables (z) used in the substitutions*/
  std::vector<VariablePtr> additionalVariables_;

  /** Other variables (y), i.e. the variables present in the constraints used
   * for the substitutions but not substituted.
   */
  std::vector<VariablePtr> otherVariables_;

  /** The additional constraints to add to the problem*/
  std::vector<std::shared_ptr<constraint::BasicLinearConstraint>> additionalConstraints_;

  friend class SubstitutionTest;
};

} // namespace internal

} // namespace hint

} // namespace tvm
