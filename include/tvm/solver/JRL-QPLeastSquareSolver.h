/* Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/solver/abstract/LeastSquareSolver.h>

#include <Eigen/QR>

#include <jrl-qp/GoldfarbIdnaniSolver.h>

namespace tvm
{

namespace solver
{
class JRLQPLSSolverFactory;

/** A set of options for JRLQPLeastSquareSolver */
class TVM_DLLAPI JRLQPLSSolverOptions
{
  TVM_ADD_NON_DEFAULT_OPTION(big_number, constant::big_number)
  TVM_ADD_NON_DEFAULT_OPTION(cholesky, false)
  TVM_ADD_NON_DEFAULT_OPTION(choleskyDamping, 1e-6)
  TVM_ADD_NON_DEFAULT_OPTION(damping, 1e-12)
  TVM_ADD_NON_DEFAULT_OPTION(verbose, false)

public:
  using Factory = JRLQPLSSolverFactory;
};

/** An encapsulation of jrl-qp's GoldfarbIdnaniSolver solver, to solve linear
 * least-squares problems.
 */
class TVM_DLLAPI JRLQPLeastSquareSolver : public abstract::LeastSquareSolver
{
public:
  JRLQPLeastSquareSolver(const JRLQPLSSolverOptions & options = {});

protected:
  void initializeBuild_(int nObj, int nEq, int nIneq, bool useBounds) override;
  ImpactFromChanges resize_(int nObj, int nEq, int nIneq, bool useBounds) override;
  void addBound_(LinearConstraintPtr bound, RangePtr range, bool first) override;
  void addEqualityConstraint_(LinearConstraintPtr cstr) override;
  void addIneqalityConstraint_(LinearConstraintPtr cstr) override;
  void addObjective_(LinearConstraintPtr cstr, SolvingRequirementsPtr req, double additionalWeight) override;
  void setMinimumNorm_() override;
  void resetBounds_() override;
  void preAssignmentProcess_() override;
  void postAssignmentProcess_() override;
  bool solve_() override;
  virtual const Eigen::VectorXd & result_() const override;
  bool handleDoubleSidedConstraint_() const override { return false; }
  Range nextEqualityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const override;
  Range nextInequalityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const override;
  Range nextObjectiveRange_(const constraint::abstract::LinearConstraint & cstr) const override;
  void removeBounds_(const Range & range) override;
  void updateEqualityTargetData(scheme::internal::AssignmentTarget & target) override;
  void updateInequalityTargetData(scheme::internal::AssignmentTarget & target) override;
  void updateBoundTargetData(scheme::internal::AssignmentTarget & target) override;
  void updateObjectiveTargetData(scheme::internal::AssignmentTarget & target) override;

  void applyImpactLogic(ImpactFromChanges & impact) override;

  void printProblemData_() const override;
  void printDiagnostic_() const override;

private:
  Eigen::MatrixXd D_; // We have Q = D^T D
  Eigen::VectorXd e_; // We have a = D^t e
  Eigen::MatrixXd Q_;
  Eigen::VectorXd a_;
  Eigen::MatrixXd C_;
  Eigen::VectorXd bl_;
  Eigen::VectorXd bu_;
  Eigen::VectorXd xl_;
  Eigen::VectorXd xu_;

  Eigen::VectorXd xStar_; //solution
  jrl::qp::TerminationStatus status_;

  jrl::qp::GoldfarbIdnaniSolver gi_;

  bool autoMinNorm_;

  // options
  double big_number_;
  double damping_;         // value added to the diagonal of Q for regularization
};

/** A factory class to create JRLQPLeastSquareSolver instances with a given
 * set of options.
 */
class TVM_DLLAPI JRLQPLSSolverFactory : public abstract::LSSolverFactory
{
public:
  std::unique_ptr<abstract::LSSolverFactory> clone() const override;

  /** Creation of a configuration from a set of options*/
  JRLQPLSSolverFactory(const JRLQPLSSolverOptions & options = {});

  std::unique_ptr<abstract::LeastSquareSolver> createSolver() const override;

private:
  JRLQPLSSolverOptions options_;
};

} // namespace solver

} // namespace tvm