/* Copyright 2022 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/solver/LexLSHierarchicalLeastSquareSolver.h>

#include <tvm/scheme/internal/AssignmentTarget.h>

#include <iostream>
#include <numeric> // for std::iota

namespace tvm
{

namespace solver
{
LexLSHierarchicalLeastSquareSolver::LexLSHierarchicalLeastSquareSolver(const LexLSHLSSolverOptions & options)
: HierarchicalLeastSquareSolver(options.verbose().value()), boundData_(1, 1), data_(), A_(), l_(), u_(),
  xl_(boundData_.col(0)), xu_(boundData_.col(0)), warmStart_(options.warmStart().value()), solver_(),
  autoMinNorm_(false), big_number_(options.big_number().value()),
  feasibleFirstLevel_(options.feasibleFirstLevel().value())
{
  LexLS::ParametersLexLSI param;
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(max_number_of_factorizations, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(max_number_of_factorizations, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(tol_linear_dependence, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(tol_wrong_sign_lambda, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(tol_correct_sign_lambda, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(tol_feasibility, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(regularization_type, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(max_number_of_CG_iterations, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(variable_regularization_factor, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(cycling_handling_enabled, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(cycling_max_counter, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(cycling_relax_step, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(output_file_name, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(modify_x_guess_enabled, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(modify_type_active_enabled, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(modify_type_inactive_enabled, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(set_min_init_ctr_violation, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(use_phase1_v0, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(log_working_set_enabled, param);
  TVM_PROCESS_OPTION_PUBLIC_ACCESS(deactivate_first_wrong_sign, param);

  solver_.setParameters(param);
}

void LexLSHierarchicalLeastSquareSolver::initializeBuild_(const std::vector<int> & nEq,
                                                          const std::vector<int> & nIneq,
                                                          bool useBounds)
{
  resize_(nEq, nIneq, useBounds);

  autoMinNorm_ = false;
}

LexLSHierarchicalLeastSquareSolver::ImpactFromChanges LexLSHierarchicalLeastSquareSolver::resize_(
    const std::vector<int> & nEq,
    const std::vector<int> & nIneq,
    bool useBounds)
{
  int n = variables().totalSize();
  int nLvl = static_cast<int>(nEq.size());
  ImpactFromChanges impact(nLvl);
  std::vector<int> nCstr(nLvl);

  // Create data for the new levels, before resizing and referencing
  impact.newLevels_ = nLvl - data_.size();
  Eigen::MatrixXd dummy(1, 1);
  for(size_t i = data_.size(); i < nLvl; ++i)
  {
    data_.emplace_back();
    A_.emplace_back(dummy.leftCols(1));
    l_.emplace_back(dummy.col(0));
    u_.emplace_back(dummy.col(0));
  }

  if(useBounds && feasibleFirstLevel_)
  {
    impact.bounds_ = ImpactFromChanges::willReallocate(boundData_, n, 2);
    boundData_.resize(n, 2);
    new(&xl_) VectorRef(boundData_.col(0));
    new(&xu_) VectorRef(boundData_.col(1));
    xl_.setConstant(-big_number_);
    xu_.setConstant(+big_number_);
  }

  x0_ = Eigen::VectorXd::Zero(n);

  for(int i = 0; i < nLvl; ++i)
  {
    nCstr[i] = nEq[i] + nIneq[i];
    if(i == 0 && useBounds && !feasibleFirstLevel_)
    {
      impact.equalityConstraints_[i] = ImpactFromChanges::willReallocate(data_[i], nCstr[i] + n, n + 2);
      data_[i].resize(nCstr[i] + n, n + 2);
      new(&xl_) VectorRef(data_[i].col(n).head(n));
      new(&xu_) VectorRef(data_[i].col(n + 1).head(n));
      data_[i].topLeftCorner(n, n).setIdentity();
      xl_.setConstant(-big_number_);
      xu_.setConstant(+big_number_);
      new(&A_[i]) MatrixRef(data_[i].leftCols(n).bottomRows(nCstr[i]));
      new(&l_[i]) VectorRef(data_[i].col(n).tail(nCstr[i]));
      new(&u_[i]) VectorRef(data_[i].col(n + 1).tail(nCstr[i]));
    }
    else
    {
      impact.equalityConstraints_[i] = ImpactFromChanges::willReallocate(data_[i], nCstr[i], n + 2);
      data_[i].resize(nCstr[i], n + 2);
      new(&A_[i]) MatrixRef(data_[i].leftCols(n));
      new(&l_[i]) VectorRef(data_[i].col(n));
      new(&u_[i]) VectorRef(data_[i].col(n + 1));
    }
    A_[i].setZero();
    l_[i].setConstant(-big_number_);
    u_[i].setConstant(+big_number_);

    impact.inequalityConstraints_[i] = impact.equalityConstraints_[i];
  }

  if(impact.any())
  {
    std::vector<LexLS::Index> objDim;
    std::vector<LexLS::ObjectiveType> objType;
    if(useBounds && feasibleFirstLevel_)
    {
      objDim.push_back(n);
      objType.push_back(LexLS::ObjectiveType::SIMPLE_BOUNDS_OBJECTIVE);
    }
    for(auto m : nCstr)
    {
      objDim.push_back(m);
      objType.push_back(LexLS::ObjectiveType::GENERAL_OBJECTIVE);
    }

    solver_.resize(n, objDim.size(), objDim.data(), objType.data());
  }

  if(varIndex_.size() != static_cast<size_t>(n))
  {
    varIndex_.resize(n);
    std::iota(varIndex_.begin(), varIndex_.end(), 0);
  }

  if(autoMinNorm_)
  {
    A_.back().setIdentity();
    l_.back().setZero();
    u_.back().setZero();
  }

  return impact;
} // namespace solver

void LexLSHierarchicalLeastSquareSolver::addBound_(LinearConstraintPtr bound, RangePtr range, bool first)
{
  scheme::internal::AssignmentTarget target(range, xl_, xu_);
  addAssignement(bound, target, bound->variables()[0], first);
}

void LexLSHierarchicalLeastSquareSolver::addEqualityConstraint_(int lvl, LinearConstraintPtr cstr)
{
  RangePtr r = std::make_shared<Range>(nextEqualityConstraintRange_(lvl, *cstr));
  scheme::internal::AssignmentTarget target(r, A_[lvl], l_[lvl], u_[lvl], constraint::RHS::AS_GIVEN);
  addAssignement(cstr, nullptr, target, variables(), substitutions());
}

void LexLSHierarchicalLeastSquareSolver::addIneqalityConstraint_(int lvl, LinearConstraintPtr cstr)
{
  RangePtr r = std::make_shared<Range>(nextInequalityConstraintRange_(lvl, *cstr));
  scheme::internal::AssignmentTarget target(r, A_[lvl], l_[lvl], u_[lvl], constraint::RHS::AS_GIVEN);
  addAssignement(cstr, nullptr, target, variables(), substitutions());
}

void LexLSHierarchicalLeastSquareSolver::setMinimumNorm_()
{
  // TODO: this could be handled by the option regularization_type of lexlsi
  autoMinNorm_ = true;
  A_.back().setIdentity();
  l_.back().setZero();
  u_.back().setZero();
}

void LexLSHierarchicalLeastSquareSolver::resetBounds_()
{
  int n = variables().totalSize();
  xl_.setConstant(-big_number_);
  xu_.setConstant(+big_number_);
}

void LexLSHierarchicalLeastSquareSolver::preAssignmentProcess_() {}

void LexLSHierarchicalLeastSquareSolver::postAssignmentProcess_()
{
  solver_.reset();
  int i0 = 0;
  if(useBounds_ && feasibleFirstLevel_)
  {
    solver_.setData(0, varIndex_.data(), boundData_);
    ++i0;
  }
  for(size_t i = 0; i < data_.size(); ++i)
    solver_.setData(static_cast<LexLS::Index>(i + i0), data_[i]);

  if(warmStart_)
  {
    for(const auto & ci : act_)
    {
      auto a = ci.ctr_type;
      if(a == LexLS::CTR_ACTIVE_LB || a == LexLS::CTR_ACTIVE_UB)
        solver_.activate(ci.obj_index, ci.ctr_index, a, false);
    }
    solver_.set_x0(x0_);
  }
}

bool LexLSHierarchicalLeastSquareSolver::solve_()
{
  auto status = solver_.solve();

  if(warmStart_)
  {
    act_.clear();
    solver_.getActiveCtr_order(act_);
    x0_ = solver_.get_x();
  }

  return status == LexLS::TerminationStatus::PROBLEM_SOLVED
         || status == LexLS::TerminationStatus::PROBLEM_SOLVED_CYCLING_HANDLING;
}

const Eigen::VectorXd & LexLSHierarchicalLeastSquareSolver::result_() const { return solver_.get_x(); }

Range LexLSHierarchicalLeastSquareSolver::nextEqualityConstraintRange_(
    int lvl,
    const constraint::abstract::LinearConstraint & cstr) const
{
  assert(eqSize_[lvl] + ineqSize_[lvl] + cstr.size() <= nEq_[lvl] + nIneq_[lvl]
         && "Not enough rows were allocated to add this constraints at this level.");
  return {eqSize_[lvl] + ineqSize_[lvl], cstr.size()};
}

Range LexLSHierarchicalLeastSquareSolver::nextInequalityConstraintRange_(
    int lvl,
    const constraint::abstract::LinearConstraint & cstr) const
{
  assert(eqSize_[lvl] + ineqSize_[lvl] + cstr.size() <= nEq_[lvl] + nIneq_[lvl]
         && "Not enough rows were allocated to add this constraints at this level.");
  return {eqSize_[lvl] + ineqSize_[lvl], cstr.size()};
}

void LexLSHierarchicalLeastSquareSolver::removeBounds_(const Range & r)
{
  xl_.segment(r.start, r.dim).setConstant(-big_number_);
  xu_.segment(r.start, r.dim).setConstant(+big_number_);
}

void LexLSHierarchicalLeastSquareSolver::updateEqualityTargetData(int lvl, scheme::internal::AssignmentTarget & target)
{
  target.changeData(A_[lvl], l_[lvl], u_[lvl]);
}

void LexLSHierarchicalLeastSquareSolver::updateInequalityTargetData(int lvl,
                                                                    scheme::internal::AssignmentTarget & target)
{
  target.changeData(A_[lvl], l_[lvl], u_[lvl]);
}

void LexLSHierarchicalLeastSquareSolver::updateBoundTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(VectorRef(xl_), xu_);
}

void LexLSHierarchicalLeastSquareSolver::applyImpactLogic(ImpactFromChanges & impact)
{
  for(size_t i = 0; i < impact.equalityConstraints_.size(); ++i)
  {
    if(impact.equalityConstraints_[i])
      impact.inequalityConstraints_[i] = true;
    if(impact.inequalityConstraints_[i])
      impact.equalityConstraints_[i] = true;
  }
}

void LexLSHierarchicalLeastSquareSolver::printProblemData_() const { solver_.print("data"); }

void LexLSHierarchicalLeastSquareSolver::printDiagnostic_() const
{
  solver_.print("nIterations");
  solver_.print("x");
  solver_.print("v");
  solver_.print("working_set");
}

LexLSHLSSolverFactory::LexLSHLSSolverFactory(const LexLSHLSSolverOptions & options)
: HLSSolverFactory("LexLS"), options_(options)
{}

std::unique_ptr<abstract::HLSSolverFactory> LexLSHLSSolverFactory::clone() const
{
  return std::make_unique<LexLSHLSSolverFactory>(*this);
}

std::unique_ptr<abstract::HierarchicalLeastSquareSolver> LexLSHLSSolverFactory::createSolver() const
{
  return std::make_unique<LexLSHierarchicalLeastSquareSolver>(options_);
}
} // namespace solver

} // namespace tvm
