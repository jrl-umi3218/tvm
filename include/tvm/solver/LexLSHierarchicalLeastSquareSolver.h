/* Copyright 2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/solver/abstract/HierarchicalLeastSquareSolver.h>

#include <lexls/lexlsi.h>

namespace tvm
{

namespace solver
{
class LexLSLSSolverFactory;

/** A set of options for LexLSHierarchicalHierarchicalLeastSquareSolver */
class TVM_DLLAPI LexLSHLSSolverOptions
{

  TVM_ADD_DEFAULT_OPTION(max_number_of_factorizations, LexLS::Index);
  TVM_ADD_DEFAULT_OPTION(tol_linear_dependence, LexLS::RealScalar);
  TVM_ADD_DEFAULT_OPTION(tol_wrong_sign_lambda, LexLS::RealScalar);
  TVM_ADD_DEFAULT_OPTION(tol_correct_sign_lambda, LexLS::RealScalar);
  TVM_ADD_DEFAULT_OPTION(tol_feasibility, LexLS::RealScalar);
  TVM_ADD_DEFAULT_OPTION(regularization_type, LexLS::RegularizationType);
  TVM_ADD_DEFAULT_OPTION(max_number_of_CG_iterations, LexLS::Index);
  TVM_ADD_DEFAULT_OPTION(variable_regularization_factor, LexLS::RealScalar);
  TVM_ADD_NON_DEFAULT_OPTION(cycling_handling_enabled, true);
  TVM_ADD_DEFAULT_OPTION(cycling_max_counter, LexLS::Index);
  TVM_ADD_DEFAULT_OPTION(cycling_relax_step, LexLS::RealScalar);
  TVM_ADD_DEFAULT_OPTION(output_file_name, std::string);
  TVM_ADD_DEFAULT_OPTION(modify_x_guess_enabled, bool);
  TVM_ADD_DEFAULT_OPTION(modify_type_active_enabled, bool);
  TVM_ADD_DEFAULT_OPTION(modify_type_inactive_enabled, bool);
  TVM_ADD_DEFAULT_OPTION(set_min_init_ctr_violation, bool);
  TVM_ADD_DEFAULT_OPTION(use_phase1_v0, bool);
  TVM_ADD_DEFAULT_OPTION(log_working_set_enabled, bool);
  TVM_ADD_DEFAULT_OPTION(deactivate_first_wrong_sign, bool);

  TVM_ADD_NON_DEFAULT_OPTION(big_number, constant::big_number)
  TVM_ADD_NON_DEFAULT_OPTION(verbose, false)
  TVM_ADD_NON_DEFAULT_OPTION(warmStart, false)

public:
  using Factory = LexLSLSSolverFactory;
};

/** An encapsulation of the LexLS solver, to solve linear least-squares problems. */
class TVM_DLLAPI LexLSHierarchicalLeastSquareSolver : public abstract::HierarchicalLeastSquareSolver
{
public:
  LexLSHierarchicalLeastSquareSolver(const LexLSHLSSolverOptions & options = {});

protected:
  void initializeBuild_(const std::vector<int> & nEq, const std::vector<int> & nIneq, bool useBounds) override;
  ImpactFromChanges resize_(const std::vector<int> & nEq, const std::vector<int> & nIneq, bool useBounds) override;
  void addBound_(LinearConstraintPtr bound, RangePtr range, bool first) override;
  void addEqualityConstraint_(int lvl, LinearConstraintPtr cstr) override;
  void addIneqalityConstraint_(int lvl, LinearConstraintPtr cstr) override;
  void setMinimumNorm_() override;
  void resetBounds_() override;
  void preAssignmentProcess_() override;
  void postAssignmentProcess_() override;
  bool solve_() override;
  const Eigen::VectorXd & result_() const override;
  bool handleDoubleSidedConstraint_() const override { return true; }
  Range nextEqualityConstraintRange_(int lvl, const constraint::abstract::LinearConstraint & cstr) const override;
  Range nextInequalityConstraintRange_(int lvl, const constraint::abstract::LinearConstraint & cstr) const override;

  void removeBounds_(const Range & range) override;
  void updateEqualityTargetData(int lvl, scheme::internal::AssignmentTarget & target) override;
  void updateInequalityTargetData(int lvl, scheme::internal::AssignmentTarget & target) override;
  void updateBoundTargetData(scheme::internal::AssignmentTarget & target) override;

  void applyImpactLogic(ImpactFromChanges & impact);

  void printProblemData_() const override;
  void printDiagnostic_() const override;

private:
  using MatrixXdCol = decltype(Eigen::MatrixXd().col(0));
  using MatrixXdBlock = decltype(Eigen::MatrixXd().leftCols(0));

  Eigen::MatrixXd boundData_;
  std::vector<Eigen::MatrixXd> data_;

  MatrixXdCol xl_;
  MatrixXdCol xu_;
  std::vector<MatrixXdBlock> A_;
  std::vector<MatrixXdCol> l_;
  std::vector<MatrixXdCol> u_; 

  std::vector<LexLS::Index> varIndex_;

  bool warmStart_;
  Eigen::VectorXd x0_;
  std::vector<LexLS::ConstraintIdentifier> act_;

  LexLS::internal::LexLSI solver_;

  bool autoMinNorm_;
  double big_number_;
};

/** A factory class to create LexLSHierarchicalLeastSquareSolver instances with a given
 * set of options.
 */
class TVM_DLLAPI LexLSHLSSolverFactory : public abstract::HLSSolverFactory
{
public:
  /** Creation of a configuration from a set of options*/
  LexLSHLSSolverFactory(const LexLSHLSSolverOptions & options = {});

  std::unique_ptr<abstract::HLSSolverFactory> clone() const override;
  std::unique_ptr<abstract::HierarchicalLeastSquareSolver> createSolver() const override;

private:
  LexLSHLSSolverOptions options_;
};

} // namespace solver

} // namespace tvm
