/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>

#include <tvm/Variable.h> // Range
#include <tvm/constraint/abstract/LinearConstraint.h>
#include <tvm/hint/internal/Substitutions.h>
#include <tvm/requirements/SolvingRequirements.h>
#include <tvm/scheme/internal/AssignmentTarget.h>
#include <tvm/scheme/internal/CompiledAssignmentWrapper.h>
#include <tvm/scheme/internal/MatrixAssignment.h>

#include <Eigen/Core>

#include <memory>
#include <type_traits>
#include <vector>

namespace tvm::scheme::internal
{

/** A class whose role is to assign efficiently the matrix and vector(s) of a
 * LinearConstraint to a part of matrix and vector(s) specified by a
 * ResolutionScheme and a mapping of variables. This is done while taking
 * into account the possible convention differences between the constraint
 * and the scheme, as well as the requirements on the constraint.
 */
class TVM_DLLAPI Assignment
{
public:
  /** Pointer type to a method of LinearConstraint returning a vector.
   * It is used to make a selection between e(), l() and u().
   */
  using RHSFunction = const Eigen::VectorXd & (constraint::abstract::LinearConstraint::*)() const;

  /** Pointer type to a method of AssignmentTarget returning a matrix block.
   * It is used to make a selection between A(), AFirstHalf() and ASecondHalf().
   */
  using MatrixFunction = MatrixRef (AssignmentTarget::*)(int, int) const;

  /** Pointer type to a method of AssignementTarget returning a vector segment.
   * It is used to make a selection between b(), bFirstHalf(), bSecondHalf(),
   * l() and u().
   */
  using VectorFunction = VectorRef (AssignmentTarget::*)() const;

  /** \private Dummy struct serving as reminder*/
  struct IWontForgetToCallUpdates
  {};

  /** Assignment constructor
   * \param source The linear constraints whose matrix and vector(s) will be
   * assigned.
   * \param req Solving requirements attached to this constraint.
   * \param target The target of the assignment.
   * \param variables The vector of variables corresponding to the target.
   * It must be such that its total dimension is equal to the column size of
   * the target matrix.
   * \param scalarizationWeight An additional scalar weight to apply on the
   * constraint, used by the solver to emulate priority.
   */
  Assignment(LinearConstraintPtr source,
             SolvingRequirementsPtr req,
             const AssignmentTarget & target,
             const VariableVector & variables,
             const hint::internal::Substitutions * const subs = nullptr,
             double scalarizationWeight = 1);

  /** Version for bounds
   * \param first whether this is the first assignment of bounds for this
   * variable (first assignment just copy vectors while the following ones
   * need to perform min/max operations).
   */
  Assignment(LinearConstraintPtr source, const AssignmentTarget & target, const VariablePtr & variables, bool first);

  Assignment(const Assignment &) = delete;
  Assignment(Assignment &&) = default;
  Assignment & operator=(const Assignment &) = delete;
  Assignment & operator=(Assignment &&) = default;

  static Assignment reprocess(const Assignment &, const VariableVector &, const hint::internal::Substitutions * const);
  static Assignment reprocess(const Assignment &, const VariablePtr &, bool);

  AssignmentTarget & target(IWontForgetToCallUpdates = {});
  /** Change the weight.*/
  bool changeScalarWeightIsAllowed();
  bool changeVectorWeightIsAllowed();

  /** To be called when the source has been resized*/
  void onUpdatedSource();
  /** To be called when the target has been resized and/or range has changed*/
  void onUpdatedTarget();
  /** To be called when the variables change.*/
  void onUpdatedMapping(const VariableVector & newVar, bool updateMatrixtarget = true);
  /** To be called after changing the weights.*/
  void onUpdateWeights(bool scalar = true, bool vector = true);

  /** Perform the assignment.*/
  void run();

  static double big_;

private:
  /** Check that the convention and size of the target are compatible with the
   * convention and size of the source.
   */
  void checkTarget();

  void checkBounds();

  /** Generates the assignments for the general case.
   * \param variables the set of variables for the problem.
   */
  void build(const VariableVector & variables);

  /** Generates the assignments for the bound case.
   * \param variables the set of variables for the problem.
   * \param first true if this is the first assignment for the bounds (first
   * assignment makes copy, the following perform min/max
   */
  void build(const VariablePtr & variable, bool first);

  /** Build internal data from the requirements*/
  void processRequirements();

  /** Creates a matrix assignment from the jacobian of \p source_ corresponding
   * to variable \p x to the block of matrix described by \p M and \p range.
   * \p flip indicates a sign change if \p true.
   */
  void addMatrixAssignment(Variable & x, MatrixFunction M, const Range & range, bool flip);

  /** Creates the assignments due to substituting the variable \p x by the
   * linear expression given by \p sub. The target is given by \p M.
   * \p flip indicates a sign change if \p true.
   */
  void addMatrixSubstitutionAssignments(const VariableVector & variables,
                                        Variable & x,
                                        MatrixFunction M,
                                        const function::BasicLinearFunction & sub,
                                        bool flip);

  /** Creates and assignment between the vector given by \p f and the one given
   * by \p v, taking care of the RHS conventions for the source and the
   * target. The assignment type is given by the template parameter \p A.
   * \p flip indicates a sign change if \p true.
   */
  template<AssignType A = AssignType::COPY, typename From, typename To>
  void addVectorAssignment(const From & f, To v, bool flip, bool useFRHS = true, bool useTRHS = true);

  /** Creates and assignment between the vector given by \p f and the one given
   * by \p v, taking care of the RHS conventions for the source and the
   * target. The source is premultiplied by the inverse of the diagonal matrix
   * \p D. The assignment type is given by the template parameter \p A.
   * \p flip indicates a sign change if \p true.
   */
  template<AssignType A = AssignType::COPY, typename From, typename To>
  void addVectorAssignment(const From & f,
                           To v,
                           const MatrixConstRef & D,
                           bool flip,
                           bool useFRHS = true,
                           bool useTRHS = true);

  /** Creates the assignments due to the substitution of variable \p x by the
   * linear expression \p sub. The target is given by \p v.
   */
  void addVectorSubstitutionAssignments(const function::BasicLinearFunction & sub,
                                        VectorFunction v,
                                        Variable & x,
                                        bool flip);

  /** Create a vector assignment where the source is a constant. The target
   * is given by \p v and the type of assignment by \p A.
   */
  template<AssignType A = AssignType::COPY, typename To>
  void addVectorAssignment(double d, To v, bool flip = false, bool useFRHS = true, bool useTRHS = true);

  /** Create a vector assignment where the source is a constant that is
   * premultiplied by the inverse of a diagonal matrix \p D.
   * The target is given by \p v and the type of assignment by \p A.
   */
  template<AssignType A = AssignType::COPY, typename To>
  void addVectorAssignment(double d,
                           To v,
                           const MatrixConstRef & D,
                           bool flip = false,
                           bool useFRHS = true,
                           bool useTRHS = true);

  /** Creates an assignment setting to zero the matrix block given by \p M
   * and \p range. The variable \p x is simply stored in the corresponding
   * \p MatrixAssignment.
   */
  void addZeroAssignment(Variable & x, MatrixFunction M, const Range & range);

  /** Calls addAssignments(const VariableVector& variables, MatrixFunction M,
   * RHSFunction f1, VectorFunction v1, RHSFunction f2, VectorFunction v2,
   * bool flip) for a single-sided case.
   */
  void addAssignments(const VariableVector & variables, MatrixFunction M, RHSFunction f, VectorFunction v, bool flip);

  /** Calls addAssignments(const VariableVector& variables, MatrixFunction M,
   * RHSFunction f1, VectorFunction v1, RHSFunction f2, VectorFunction v2,
   * bool flip) for a double-sided case.
   */
  void addAssignments(const VariableVector & variables,
                      MatrixFunction M,
                      RHSFunction f1,
                      VectorFunction v1,
                      RHSFunction f2,
                      VectorFunction v2);

  /** Creates all the matrix assignments between the source and the target,
   * as well as the vector assignments described by \p f1 and \p v1 and
   * optionally by \p f2 and \p v2 if those are not \p nullptr.
   * This method is called after the constraint::Type conventions of the
   * source and the target have been processed (resulting in the choice of
   * \p M, \p f1, \p v1, \p f2, \p v2 and \p flip). It handles internally the
   * substitutions.
   */
  void addAssignments(const VariableVector & variables,
                      MatrixFunction M,
                      RHSFunction f1,
                      VectorFunction v1,
                      RHSFunction f2,
                      VectorFunction v2,
                      bool flip);

  void addBound(const VariablePtr & variable, RHSFunction f, bool first);

  template<typename L, typename U>
  void addBounds(const VariablePtr & variable, L l, U u, bool first);

  template<typename L, typename U, typename TL, typename TU>
  void addBounds(const VariablePtr & variable, L l, U u, TL tl, TU tu, bool first);

  /** Create the compiled assignment between \p from and \p to, taking into
   * account the requirements and the possible sign flip indicated by
   * \p flip.
   */
  template<typename T, AssignType A, typename U>
  CompiledAssignmentWrapper<T> createAssignment(const U & from, const Eigen::Ref<T> & to, bool flip = false);

  /** Create the compiled substitution assignment to = Mult * from (vector
   * case) or to = from * mult (matrix case) taking into account the
   * requirements and \p flip
   */
  template<typename T, AssignType A, MatrixMult M = GENERAL, typename U, typename V>
  CompiledAssignmentWrapper<T> createMultiplicationAssignment(const U & from,
                                                              const Eigen::Ref<T> & to,
                                                              const V & Mult,
                                                              bool flip = false);

  /** The source of the assignment.*/
  LinearConstraintPtr source_;
  /** The target of the assignment.*/
  AssignmentTarget target_;
  /** The weight used to emulate hierarchy in a weight scheme.*/
  double scalarizationWeight_;
  /** The requirements attached to the source.*/
  SolvingRequirementsPtr requirements_;
  /** Indicates if the requirements use a default weight AND the scalarizationWeight is 1.*/
  bool useDefaultScalarWeight_;
  /** Indicates if the requirements use a default anisotropic weight.*/
  bool useDefaultAnisotropicWeight_;
  /** All the assignments that are setting the initial values of the targeted blocks*/
  std::vector<MatrixAssignment> matrixAssignments_;
  /** All assignments due to substitutions. We separate them from matrixAssignments_
   * because these assignments add to existing values, and we need to be sure
   * that the assignments in matrixAssignments_ have been carried out before.
   */
  std::vector<MatrixAssignment> matrixSubstitutionAssignments_;
  /** All the initial rhs assignments*/
  std::vector<VectorAssignment> vectorAssignments_;
  /** The additional rhs assignments due to substitutions. As for matrix
   * assignments, they need to be carried out after those of vectorAssignments_.
   */
  std::vector<VectorSubstitutionAssignement> vectorSubstitutionAssignments_;

  /** Data for substitutions */
  VariableVector substitutedVariables_;
  std::vector<std::shared_ptr<function::BasicLinearFunction>> variableSubstitutions_;

  /** Helper structure grouping data whose address should remain constant through
   * a move
   */
  struct ReferenceableData
  {
    /** Processed requirements*/
    double scalarWeight_ = 1;
    double minusScalarWeight_ = -1;
    Eigen::VectorXd anisotropicWeight_;
    Eigen::VectorXd minusAnisotropicWeight_;

    /** Temporary vectors for bound assignments*/
    Eigen::VectorXd tmp1_;
    Eigen::VectorXd tmp2_;
    Eigen::VectorXd tmp3_;
    Eigen::VectorXd tmp4_;
  };

  std::unique_ptr<ReferenceableData> data_;
};

} // namespace tvm::scheme::internal

#include <tvm/scheme/internal/Assignment.hpp>
