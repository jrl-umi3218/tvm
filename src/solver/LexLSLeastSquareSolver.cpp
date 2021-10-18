/* Copyright 2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/solver/LexLSLeastSquareSolver.h>

#include <tvm/scheme/internal/AssignmentTarget.h>

#include <iostream>
#include <numeric> // for std::iota

namespace tvm
{

namespace solver
{
LexLSLeastSquareSolver::LexLSLeastSquareSolver(const LexLSLSSolverOptions & options)
: LeastSquareSolver(options.verbose().value()), data0_(1,1), data1_(1,1), data2_(1,1), A1_(data1_.leftCols(0)), l1_(data1_.col(0)), u1_(data1_.col(0)),
  A2_(data2_.leftCols(0)), l2_(data2_.col(0)), u2_(data2_.col(0)), xl_(data0_.col(0)), xu_(data0_.col(0)),
  solver_(), autoMinNorm_(false), big_number_(options.big_number().value())
{
  LexLS::ParametersLexLSI param;
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(max_number_of_factorizations, param)
}

void LexLSLeastSquareSolver::initializeBuild_(int nObj, int nEq, int nIneq, bool)
{
  resize_(nObj, nEq, nIneq, true);

  autoMinNorm_ = false;
}

LexLSLeastSquareSolver::ImpactFromChanges LexLSLeastSquareSolver::resize_(int nObj, int nEq, int nIneq, bool)
{
  int n = variables().totalSize();
  int nCstr = nEq + nIneq;
  ImpactFromChanges impact;

  impact.objectives_ = ImpactFromChanges::willReallocate(data2_, nObj, n + 2);
  data2_.resize(nObj, n + 2);
  data2_.setZero();
  new(&A2_) MatrixXdBlock(data2_.leftCols(n));
  new(&l2_) MatrixXdCol(data2_.col(n));
  new(&u2_) MatrixXdCol(data2_.col(n + 1));
  if(autoMinNorm_)
  {
    A2_.setIdentity();
    l2_.setZero();
    u2_.setZero();
  }
  else
  {
    A2_.setZero();
    l2_.setConstant(-big_number_);
    u2_.setConstant(+big_number_);
  }
  impact.equalityConstraints_ = ImpactFromChanges::willReallocate(data1_, nCstr, n + 2);
  data1_.resize(nCstr, n + 2);
  new(&A1_) MatrixXdBlock(data1_.leftCols(n));
  new(&l1_) MatrixXdCol(data1_.col(n));
  new(&u1_) MatrixXdCol(data1_.col(n + 1));
  A1_.setZero();
  l1_.setConstant(-big_number_);
  u1_.setConstant(+big_number_);
  impact.bounds_ = ImpactFromChanges::willReallocate(data0_, n, 2);
  data0_.resize(n, 2);
  new(&xl_) MatrixXdCol(data0_.col(0));
  new(&xu_) MatrixXdCol(data0_.col(1));
  xl_.setConstant(-big_number_);
  xu_.setConstant(+big_number_);

  impact.inequalityConstraints_ = impact.equalityConstraints_;
  if(impact.any())
  {
    LexLS::Index objDim[3] = {n, nCstr, nObj};
    LexLS::ObjectiveType objType[3] = {LexLS::ObjectiveType::SIMPLE_BOUNDS_OBJECTIVE,
                                       LexLS::ObjectiveType::GENERAL_OBJECTIVE,
                                       LexLS::ObjectiveType::GENERAL_OBJECTIVE};
    solver_.resize(n, 3, objDim, objType);
  }
  if(varIndex_.size() != static_cast<size_t>(n))
  {
    varIndex_.resize(n);
    std::iota(varIndex_.begin(), varIndex_.end(), 0);
  }

  return impact;
} // namespace solver

void LexLSLeastSquareSolver::addBound_(LinearConstraintPtr bound, RangePtr range, bool first)
{
  scheme::internal::AssignmentTarget target(range, xl_, xu_);
  addAssignement(bound, target, bound->variables()[0], first);
}

void LexLSLeastSquareSolver::addEqualityConstraint_(LinearConstraintPtr cstr)
{
  RangePtr r = std::make_shared<Range>(nextEqualityConstraintRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, A1_, l1_, u1_, constraint::RHS::AS_GIVEN);
  addAssignement(cstr, nullptr, target, variables(), substitutions());
}

void LexLSLeastSquareSolver::addIneqalityConstraint_(LinearConstraintPtr cstr)
{
  RangePtr r = std::make_shared<Range>(nextInequalityConstraintRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, A1_, l1_, u1_, constraint::RHS::AS_GIVEN);
  addAssignement(cstr, nullptr, target, variables(), substitutions());
}

void LexLSLeastSquareSolver::addObjective_(LinearConstraintPtr cstr,
                                           SolvingRequirementsPtr req,
                                           double additionalWeight)
{
  RangePtr r = std::make_shared<Range>(nextObjectiveRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, A2_, l2_, u2_, constraint::RHS::AS_GIVEN);
  addAssignement(cstr, req, target, variables(), substitutions(), additionalWeight);
}

void LexLSLeastSquareSolver::setMinimumNorm_()
{
  autoMinNorm_ = true;
  A2_.setIdentity();
  l2_.setZero();
  u2_.setZero();
}

void LexLSLeastSquareSolver::resetBounds_()
{
  int n = variables().totalSize();
  xl_.setConstant(-big_number_);
  xu_.setConstant(+big_number_);
}

void LexLSLeastSquareSolver::preAssignmentProcess_() {}

void LexLSLeastSquareSolver::postAssignmentProcess_() {}

bool LexLSLeastSquareSolver::solve_()
{
  solver_.setData(0, varIndex_.data(), data0_);
  solver_.setData(1, data1_);
  solver_.setData(2, data2_);
  auto status = solver_.solve();

  return status == LexLS::TerminationStatus::PROBLEM_SOLVED
         || status == LexLS::TerminationStatus::PROBLEM_SOLVED_CYCLING_HANDLING;
}

const Eigen::VectorXd & LexLSLeastSquareSolver::result_() const { return solver_.get_x(); }

Range LexLSLeastSquareSolver::nextEqualityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const
{
  return {eqSize_ + ineqSize_, cstr.size()};
}

Range LexLSLeastSquareSolver::nextInequalityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const
{
  return {eqSize_ + ineqSize_, cstr.size()};
}

Range LexLSLeastSquareSolver::nextObjectiveRange_(const constraint::abstract::LinearConstraint & cstr) const
{
  return {objSize_, cstr.size()};
}

void LexLSLeastSquareSolver::removeBounds_(const Range & r)
{
  xl_.segment(r.start, r.dim).setConstant(-big_number_);
  xu_.segment(r.start, r.dim).setConstant(+big_number_);
}

void LexLSLeastSquareSolver::updateEqualityTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(A1_, l1_, u1_);
}

void LexLSLeastSquareSolver::updateInequalityTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(A1_, l1_, u1_);
}

void LexLSLeastSquareSolver::updateBoundTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(VectorRef(xl_), xu_);
}

void LexLSLeastSquareSolver::updateObjectiveTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(A2_, l2_, u2_);
}

void LexLSLeastSquareSolver::applyImpactLogic(ImpactFromChanges & impact)
{
  if(impact.equalityConstraints_)
    impact.inequalityConstraints_ = true;
  if(impact.inequalityConstraints_)
    impact.equalityConstraints_ = true;
}

void LexLSLeastSquareSolver::printProblemData_() const { solver_.print("data"); }

void LexLSLeastSquareSolver::printDiagnostic_() const
{
  solver_.print("nIterations");
  solver_.print("x");
  solver_.print("v");
  solver_.print("working_set");
}

LexLSLSSolverFactory::LexLSLSSolverFactory(const LexLSLSSolverOptions & options)
: LSSolverFactory("LexLS"), options_(options)
{}

std::unique_ptr<abstract::LSSolverFactory> LexLSLSSolverFactory::clone() const
{
  return std::make_unique<LexLSLSSolverFactory>(*this);
}

std::unique_ptr<abstract::LeastSquareSolver> LexLSLSSolverFactory::createSolver() const
{
  return std::make_unique<LexLSLeastSquareSolver>(options_);
}
} // namespace solver

} // namespace tvm
