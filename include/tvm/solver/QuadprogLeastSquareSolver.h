/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/solver/abstract/LeastSquareSolver.h>

#include <Eigen/QR>

#include <eigen-quadprog/QuadProg.h>

namespace tvm
{

namespace solver
{
class QuadprogLSSolverFactory;

/** A set of options for QuadprogLeastSquareSolver */
class TVM_DLLAPI QuadprogLSSolverOptions
{
  TVM_ADD_NON_DEFAULT_OPTION(big_number, constant::big_number)
  TVM_ADD_NON_DEFAULT_OPTION(cholesky, false)
  TVM_ADD_NON_DEFAULT_OPTION(choleskyDamping, 1e-6)
  TVM_ADD_NON_DEFAULT_OPTION(damping, 1e-12)
  TVM_ADD_NON_DEFAULT_OPTION(verbose, false)

public:
  using Factory = QuadprogLSSolverFactory;
};

/** An encapsulation of the Quadprog solver, to solve linear least-squares problems. */
class TVM_DLLAPI QuadprogLeastSquareSolver : public abstract::LeastSquareSolver
{
public:
  QuadprogLeastSquareSolver(const QuadprogLSSolverOptions & options = {});

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
  using VectorXdSeg = decltype(Eigen::VectorXd().segment(0, 1));
  using MatrixXdRows = decltype(Eigen::MatrixXd().middleRows(0, 1));

  Eigen::MatrixXd D_; // We have Q = D^T D
  Eigen::VectorXd e_; // We have c = D^t e
  Eigen::MatrixXd Q_;
  Eigen::VectorXd c_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;

  MatrixXdRows Aineq_; // part of A_ corresponding to inequality constraints
  VectorXdSeg bineq_;  // part of b_ corresponding to inequality constraints
  VectorXdSeg xl_;     // part of b_ corresponding to lower bound constraints
  VectorXdSeg xu_;     // part of b_ corresponding to upper bound constraints

  Eigen::QuadProgDense qpd_;
  Eigen::HouseholderQR<Eigen::MatrixXd> qr_; // TODO add option for ColPiv variant

  bool autoMinNorm_;
  bool underspecifiedObj_; // true when nObj<n
  int nIneqInclBounds_;    // number of inequality constraints including bounds.

  // options
  double big_number_;
  double damping_;         // value added to the diagonal of Q for regularization (non Cholesky case)
  bool cholesky_;          // compute the Cholesky decomposition before calling the solver.
  double choleskyDamping_; // if nObj<n, the cholesky factor R is trapezoidal. A multiple of
                           // the identity is used to make it triangular using this value.
};

/** A factory class to create QuadprogLeastSquareSolver instances with a given
 * set of options.
 */
class TVM_DLLAPI QuadprogLSSolverFactory : public abstract::LSSolverFactory
{
public:
  std::unique_ptr<abstract::LSSolverFactory> clone() const override;

  /** Creation of a configuration from a set of options*/
  QuadprogLSSolverFactory(const QuadprogLSSolverOptions & options = {});

  std::unique_ptr<abstract::LeastSquareSolver> createSolver() const override;

private:
  QuadprogLSSolverOptions options_;
};

} // namespace solver

} // namespace tvm
