/* Copyright 2021-2022 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/solver/LexLSSolverOptions.h>
#include <tvm/solver/abstract/LeastSquareSolver.h>

#include <lexls/lexlsi.h>

namespace tvm::solver
{
class LexLSLSSolverFactory;

/** A set of options for LexLSLeastSquareSolver */
class TVM_DLLAPI LexLSLSSolverOptions : public LexLSSolverOptions
{
  TVM_ADD_NON_DEFAULT_OPTION(big_number, constant::big_number)
  TVM_ADD_NON_DEFAULT_OPTION(verbose, false)
  TVM_ADD_NON_DEFAULT_OPTION(warmStart, false)

public:
  using Factory = LexLSLSSolverFactory;
};

/** An encapsulation of the LexLS solver, to solve linear least-squares problems. */
class TVM_DLLAPI LexLSLeastSquareSolver : public abstract::LeastSquareSolver
{
public:
  LexLSLeastSquareSolver(const LexLSLSSolverOptions & options = {});

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
  const Eigen::VectorXd & result_() const override;
  bool handleDoubleSidedConstraint_() const override { return true; }
  Range nextEqualityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const override;
  Range nextInequalityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const override;
  Range nextObjectiveRange_(const constraint::abstract::LinearConstraint & cstr) const override;

  void removeBounds_(const Range & range) override;
  void updateEqualityTargetData(scheme::internal::AssignmentTarget & target) override;
  void updateInequalityTargetData(scheme::internal::AssignmentTarget & target) override;
  void updateBoundTargetData(scheme::internal::AssignmentTarget & target) override;
  void updateObjectiveTargetData(scheme::internal::AssignmentTarget & target) override;

  void applyImpactLogic(ImpactFromChanges & impact);

  void printProblemData_() const override;
  void printDiagnostic_() const override;

private:
  using MatrixXdCol = decltype(Eigen::MatrixXd().col(0));
  using MatrixXdBlock = decltype(Eigen::MatrixXd().leftCols(0));

  Eigen::MatrixXd data0_;
  Eigen::MatrixXd data1_;
  Eigen::MatrixXd data2_;

  MatrixXdCol xl_;
  MatrixXdCol xu_;
  MatrixXdBlock A1_;
  MatrixXdCol l1_;
  MatrixXdCol u1_;
  MatrixXdBlock A2_;
  MatrixXdCol l2_;
  MatrixXdCol u2_;

  std::vector<LexLS::Index> varIndex_;

  bool warmStart_;
  Eigen::VectorXd x0_;
  std::vector<LexLS::ConstraintActivationType> act0_;
  std::vector<LexLS::ConstraintActivationType> act1_;

  LexLS::internal::LexLSI solver_;

  bool autoMinNorm_;
  double big_number_;
};

/** A factory class to create LexLSLeastSquareSolver instances with a given
 * set of options.
 */
class TVM_DLLAPI LexLSLSSolverFactory : public abstract::LSSolverFactory
{
public:
  /** Creation of a configuration from a set of options*/
  LexLSLSSolverFactory(const LexLSLSSolverOptions & options = {});

  std::unique_ptr<abstract::LSSolverFactory> clone() const override;
  std::unique_ptr<abstract::LeastSquareSolver> createSolver() const override;

private:
  LexLSLSSolverOptions options_;
};

} // namespace tvm::solver
