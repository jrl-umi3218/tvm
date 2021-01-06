/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/solver/abstract/LeastSquareSolver.h>

#include <eigen-qld/QLDDirect.h>

#include <Eigen/QR>

namespace tvm
{

namespace solver
{
class QLDLSSolverFactory;

/** A set of options for QLDLeastSquareSolver */
class TVM_DLLAPI QLDLSSolverOptions
{
  TVM_ADD_NON_DEFAULT_OPTION(big_number, constant::big_number)
  TVM_ADD_NON_DEFAULT_OPTION(cholesky, false)
  TVM_ADD_NON_DEFAULT_OPTION(choleskyDamping, 1e-8)
  TVM_ADD_NON_DEFAULT_OPTION(eps, 1e-6)
  TVM_ADD_NON_DEFAULT_OPTION(verbose, false)
public:
  using Factory = QLDLSSolverFactory;
};

/** An encapsulation of the QLD solver, to solve linear least-squares problems. */
class TVM_DLLAPI QLDLeastSquareSolver : public abstract::LeastSquareSolver
{
public:
  QLDLeastSquareSolver(const QLDLSSolverOptions & options = {});

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
  void removeBounds_(const Range & r) override;
  void updateEqualityTargetData(scheme::internal::AssignmentTarget & target) override;
  void updateInequalityTargetData(scheme::internal::AssignmentTarget & target) override;
  void updateBoundTargetData(scheme::internal::AssignmentTarget & target) override;
  void updateObjectiveTargetData(scheme::internal::AssignmentTarget & target) override;

  void applyImpactLogic(ImpactFromChanges & impact) override;

  void printProblemData_() const override;
  void printDiagnostic_() const override;

private:
  using VectorXdTail = decltype(Eigen::VectorXd().tail(1));
  using MatrixXdBottom = decltype(Eigen::MatrixXd().bottomRows(1));

  Eigen::MatrixXd D_; // We have Q = D^T D
  Eigen::VectorXd e_; // We have c = D^t e
  Eigen::MatrixXd Q_;
  Eigen::VectorXd c_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  Eigen::VectorXd xl_;
  Eigen::VectorXd xu_;

  MatrixXdBottom Aineq_; // part of A_ corresponding to inequality constraints
  VectorXdTail bineq_;   // part of b_ corresponding to inequality constraints

  Eigen::QLDDirect qld_;
  Eigen::HouseholderQR<Eigen::MatrixXd> qr_; // TODO add option for ColPiv variant

  bool autoMinNorm_;
  bool underspecifiedObj_; // true when nObj<n

  // options
  double big_number_;
  double eps_;
  bool cholesky_;          // compute the Cholesky decomposition before calling the solver.
  double choleskyDamping_; // if nObj<n, the cholesky factor R is trapezoidal. A multiple of
                           // the identity is used to make it triangular using this value.
};

/** A factory class to create QLDLeastSquareSolver instances with a given
 * set of options.
 */
class TVM_DLLAPI QLDLSSolverFactory : public abstract::LSSolverFactory
{
public:
  std::unique_ptr<abstract::LSSolverFactory> clone() const override;

  /** Creation of a configuration from a set of options*/
  QLDLSSolverFactory(const QLDLSSolverOptions & options = {});

  std::unique_ptr<abstract::LeastSquareSolver> createSolver() const override;

private:
  QLDLSSolverOptions options_;
};

} // namespace solver

} // namespace tvm
