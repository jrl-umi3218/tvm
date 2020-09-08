/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Variable.h> // Range
#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/scheme/internal/AssignmentTarget.h>
#include <tvm/scheme/internal/CompiledAssignmentWrapper.h>

#include <Eigen/Core>

namespace tvm::scheme::internal
{
/** A structure grouping a matrix assignment and some of the elements that
 * defined it.
 */
class TVM_DLLAPI MatrixAssignment
{
public:
  /** Pointer type to a method of AssignmentTarget returning a matrix block.
   * It is used to make a selection between A(), AFirstHalf() and ASecondHalf().
   */
  using MatrixFunction = MatrixRef (AssignmentTarget::*)(int, int) const;

  CompiledAssignmentWrapper<Eigen::MatrixXd> assignment;
  Variable * x;
  Range colRange;
  MatrixFunction getTargetMatrix;

  /** Effectively update the output matrix of the underlying compiled assignment. */
  void updateTarget(const AssignmentTarget & target);
  /** Update the the column mapping of this assignment, based on the new variable
   * layout specified by \p newVar.
   * If \c updateMatrixTarget is \c true, the output matrix is effectively
   * updated (if not, only \c colRange is changed, but the change is not
   * reflected in the actual compiled assignment).
   */
  void updateMapping(const VariableVector & newVar, const AssignmentTarget & target, bool updateMatrixTarget);
};

/** A structure grouping a vector assignment and its target, for the case of
 * substitutions.
 */
class TVM_DLLAPI VectorSubstitutionAssignement
{
public:
  /** Pointer type to a method of AssignementTarget returning a vector segment.
   * It is used to make a selection between b(), bFirstHalf(), bSecondHalf(),
   * l() and u().
   */
  using VectorFunction = VectorRef (AssignmentTarget::*)() const;

  VectorSubstitutionAssignement(const CompiledAssignmentWrapper<Eigen::VectorXd> & a, VectorFunction getTarget)
  : assignment(a), getTargetVector(getTarget)
  {}

  CompiledAssignmentWrapper<Eigen::VectorXd> assignment; // The underlying assignment
  VectorFunction getTargetVector;                        // The way to retrieve the target vector part
};

/** A structure grouping a vector assignment and some of the elements that
 * defined it.
 */
class TVM_DLLAPI VectorAssignment : public VectorSubstitutionAssignement
{
public:
  /** Pointer type to a method of LinearConstraint returning a vector.
   * It is used to make a selection between e(), l() and u().
   */
  using RHSFunction = const Eigen::VectorXd & (constraint::abstract::LinearConstraint::*)() const;
  using VectorSubstitutionAssignement::VectorFunction;

  VectorAssignment(const CompiledAssignmentWrapper<Eigen::VectorXd> & a,
                   bool useSource,
                   RHSFunction getSource,
                   VectorFunction getTarget)
  : VectorSubstitutionAssignement(a, getTarget), useSource(useSource), getSourceVector(getSource)
  {}

  bool useSource;              // Whether or not this assignment uses a source
  RHSFunction getSourceVector; // The way to retrieve the source vector.
};
} // namespace tvm::scheme::internal
