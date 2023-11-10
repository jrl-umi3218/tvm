/* Copyright 2022 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/solver/LexLSSolverOptions.h>
#include <tvm/solver/abstract/HierarchicalLeastSquareSolver.h>

#include <lexls/lexlsi.h>

namespace tvm::solver
{
class LexLSHLSSolverFactory;

/** A set of options for LexLSHierarchicalLeastSquareSolver */
class TVM_DLLAPI LexLSHLSSolverOptions : public LexLSSolverOptions
{
  TVM_ADD_NON_DEFAULT_OPTION(big_number, constant::big_number)
  TVM_ADD_NON_DEFAULT_OPTION(verbose, false)
  TVM_ADD_NON_DEFAULT_OPTION(warmStart, false)
  /** If \a true, the first level will be considered as feasible, and bounds at this level will be
   * handled separately (and more efficiently).
   */
  TVM_ADD_NON_DEFAULT_OPTION(feasibleFirstLevel, false)

public:
  using Factory = LexLSHLSSolverFactory;
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
  void addEqualityConstraint_(LinearConstraintPtr cstr, SolvingRequirementsPtr req) override;
  void addIneqalityConstraint_(LinearConstraintPtr cstr, SolvingRequirementsPtr req) override;
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

  void applyImpactLogic(ImpactFromChanges & impact) override;

  void printProblemData_() const override;
  void printDiagnostic_() const override;

private:
  Eigen::MatrixXd boundData_;
  std::vector<Eigen::MatrixXd> data_;

  VectorRef xl_;
  VectorRef xu_;
  std::vector<MatrixRef> A_;
  std::vector<VectorRef> l_;
  std::vector<VectorRef> u_;

  std::vector<LexLS::Index> varIndex_;

  bool warmStart_;
  Eigen::VectorXd x0_;
  std::vector<LexLS::ConstraintIdentifier> act_;

  LexLS::internal::LexLSI solver_;

  bool autoMinNorm_;
  double big_number_;
  bool feasibleFirstLevel_;
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

} // namespace tvm::solver
