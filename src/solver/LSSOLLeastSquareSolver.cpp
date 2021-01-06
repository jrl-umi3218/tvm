/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/solver/LSSOLLeastSquareSolver.h>

#include <tvm/scheme/internal/AssignmentTarget.h>

#include <iostream>

namespace tvm
{

namespace solver
{
LSSOLLeastSquareSolver::LSSOLLeastSquareSolver(const LSSOLLSSolverOptions & options)
: LeastSquareSolver(options.verbose().value()), cl_(l_.tail(0)), cu_(u_.tail(0)),
  big_number_(options.big_number().value()), autoMinNorm_(false)
{
  TVM_PROCESS_OPTION(crashTol, ls_)
  TVM_PROCESS_OPTION(feasibilityMaxIter, ls_)
  TVM_PROCESS_OPTION(feasibilityTol, ls_)
  TVM_PROCESS_OPTION(infiniteBnd, ls_)
  TVM_PROCESS_OPTION(infiniteStep, ls_)
  TVM_PROCESS_OPTION(optimalityMaxIter, ls_)
  TVM_PROCESS_OPTION(persistence, ls_)
  TVM_PROCESS_OPTION(printLevel, ls_)
  TVM_PROCESS_OPTION(rankTol, ls_)
  TVM_PROCESS_OPTION(warm, ls_)
  TVM_PROCESS_OPTION(crashTol, fp_)
  TVM_PROCESS_OPTION(feasibilityMaxIter, fp_)
  TVM_PROCESS_OPTION(feasibilityTol, fp_)
  TVM_PROCESS_OPTION(infiniteBnd, fp_)
  TVM_PROCESS_OPTION(infiniteStep, fp_)
  TVM_PROCESS_OPTION(optimalityMaxIter, fp_)
  TVM_PROCESS_OPTION(persistence, fp_)
  TVM_PROCESS_OPTION(printLevel, fp_)
  TVM_PROCESS_OPTION(rankTol, fp_)
  TVM_PROCESS_OPTION(warm, fp_)
}

void LSSOLLeastSquareSolver::initializeBuild_(int nObj, int nEq, int nIneq, bool)
{
  resize_(nObj, nEq, nIneq, true);

  autoMinNorm_ = false;
}

LSSOLLeastSquareSolver::ImpactFromChanges LSSOLLeastSquareSolver::resize_(int nObj, int nEq, int nIneq, bool)
{
  int n = variables().totalSize();
  int nCstr = nEq + nIneq;
  ImpactFromChanges impact;

  impact.objectives_ = ImpactFromChanges::willReallocate(A_, nObj, n);
  A_.resize(nObj, n);
  A_.setZero();
  b_.resize(nObj);
  b_.setZero();
  impact.equalityConstraints_ = ImpactFromChanges::willReallocate(C_, nCstr, n);
  C_.resize(nCstr, n);
  C_.setZero();
  impact.bounds_ = ImpactFromChanges::willReallocate(l_, nCstr + n);
  l_ = Eigen::VectorXd::Constant(nCstr + n, -big_number_);
  u_ = Eigen::VectorXd::Constant(nCstr + n, +big_number_);
  new(&cl_) VectorXdTail(l_.tail(nCstr));
  new(&cu_) VectorXdTail(u_.tail(nCstr));
  if(nObj > 0)
    ls_.resize(n, nCstr, Eigen::lssol::eType::LS1);
  else
    fp_.resize(n, nCstr);

  impact.inequalityConstraints_ = impact.equalityConstraints_;
  impact.bounds_ = impact.bounds_ || impact.inequalityConstraints_;
  return impact;
}

void LSSOLLeastSquareSolver::addBound_(LinearConstraintPtr bound, RangePtr range, bool first)
{
  scheme::internal::AssignmentTarget target(range, l_, u_);
  addAssignement(bound, target, bound->variables()[0], first);
}

void LSSOLLeastSquareSolver::addEqualityConstraint_(LinearConstraintPtr cstr)
{
  RangePtr r = std::make_shared<Range>(nextEqualityConstraintRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, C_, cl_, cu_, constraint::RHS::AS_GIVEN);
  addAssignement(cstr, nullptr, target, variables(), substitutions());
}

void LSSOLLeastSquareSolver::addIneqalityConstraint_(LinearConstraintPtr cstr)
{
  RangePtr r = std::make_shared<Range>(nextInequalityConstraintRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, C_, cl_, cu_, constraint::RHS::AS_GIVEN);
  addAssignement(cstr, nullptr, target, variables(), substitutions());
}

void LSSOLLeastSquareSolver::addObjective_(LinearConstraintPtr cstr,
                                           SolvingRequirementsPtr req,
                                           double additionalWeight)
{
  RangePtr r = std::make_shared<Range>(nextObjectiveRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, A_, b_, constraint::Type::EQUAL, constraint::RHS::AS_GIVEN);
  addAssignement(cstr, req, target, variables(), substitutions(), additionalWeight);
}

void LSSOLLeastSquareSolver::setMinimumNorm_()
{
  autoMinNorm_ = true;
  b_.setZero();
}

void LSSOLLeastSquareSolver::resetBounds_()
{
  int n = variables().totalSize();
  l_.head(n).setConstant(-big_number_);
  u_.head(n).setConstant(+big_number_);
}

void LSSOLLeastSquareSolver::preAssignmentProcess_()
{
  // LSSOL is overwriting A during the resolution.
  // We need to make sure that A is clean before assignments are carried out.
  if(!autoMinNorm_)
    A_.setZero();
}

void LSSOLLeastSquareSolver::postAssignmentProcess_()
{
  if(autoMinNorm_)
    A_.setIdentity();
}

bool LSSOLLeastSquareSolver::solve_()
{
  if(nObj_ > 0)
    return ls_.solve(A_, b_, C_, l_, u_);
  else
    return fp_.solve(C_, l_, u_);
}

const Eigen::VectorXd & LSSOLLeastSquareSolver::result_() const
{
  if(nObj_ > 0)
    return ls_.result();
  else
    return fp_.result();
}

Range LSSOLLeastSquareSolver::nextEqualityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const
{
  return {eqSize_ + ineqSize_, cstr.size()};
}

Range LSSOLLeastSquareSolver::nextInequalityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const
{
  return {eqSize_ + ineqSize_, cstr.size()};
}

Range LSSOLLeastSquareSolver::nextObjectiveRange_(const constraint::abstract::LinearConstraint & cstr) const
{
  return {objSize_, cstr.size()};
}

void LSSOLLeastSquareSolver::removeBounds_(const Range & r)
{
  l_.segment(r.start, r.dim).setConstant(-big_number_);
  u_.segment(r.start, r.dim).setConstant(+big_number_);
}

void LSSOLLeastSquareSolver::updateEqualityTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(C_, cl_, cu_);
}

void LSSOLLeastSquareSolver::updateInequalityTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(C_, cl_, cu_);
}

void LSSOLLeastSquareSolver::updateBoundTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(VectorRef(l_), u_);
}

void LSSOLLeastSquareSolver::updateObjectiveTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(MatrixRef(A_), b_);
}

void LSSOLLeastSquareSolver::applyImpactLogic(ImpactFromChanges & impact)
{
  if(impact.equalityConstraints_)
    impact.inequalityConstraints_ = true;
  if(impact.inequalityConstraints_)
    impact.equalityConstraints_ = true;
}

void LSSOLLeastSquareSolver::printProblemData_() const
{
  std::cout << "A =\n" << A_ << std::endl;
  std::cout << "b = " << b_.transpose() << std::endl;
  std::cout << "C =\n" << C_ << std::endl;
  std::cout << "l = " << l_.transpose() << std::endl;
  std::cout << "u = " << u_.transpose() << std::endl;
}

void LSSOLLeastSquareSolver::printDiagnostic_() const
{
  if(nObj_)
  {
    std::cout << ls_.inform() << std::endl;
    ls_.print_inform();
  }
  else
  {
    std::cout << fp_.inform() << std::endl;
    fp_.print_inform();
  }
}

LSSOLLSSolverFactory::LSSOLLSSolverFactory(const LSSOLLSSolverOptions & options)
: LSSolverFactory("lssol"), options_(options)
{}

std::unique_ptr<abstract::LSSolverFactory> LSSOLLSSolverFactory::clone() const
{
  return std::make_unique<LSSOLLSSolverFactory>(*this);
}

std::unique_ptr<abstract::LeastSquareSolver> LSSOLLSSolverFactory::createSolver() const
{
  return std::make_unique<LSSOLLeastSquareSolver>(options_);
}
} // namespace solver

} // namespace tvm