/* Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/solver/JRL-QPLeastSquareSolver.h>

#include <tvm/scheme/internal/AssignmentTarget.h>

#include <iostream>

namespace
{
// TODO: should it be moved to jrl-qp ?
std::ostream & operator<<(std::ostream & os, jrl::qp::TerminationStatus s)
{
  switch(s)
  {
    case jrl::qp::TerminationStatus::SUCCESS:
      os << "success";
      return os;
    case jrl::qp::TerminationStatus::INCONSISTENT_INPUT:
      os << "inconsistent inputs";
      return os;
    case jrl::qp::TerminationStatus::NON_POS_HESSIAN:
      os << "non positive hessian matrix";
      return os;
    case jrl::qp::TerminationStatus::INFEASIBLE:
      os << "infeasible";
      return os;
    case jrl::qp::TerminationStatus::MAX_ITER_REACHED:
      os << "maximum iteration reached";
      return os;
    case jrl::qp::TerminationStatus::LINEAR_DEPENDENCY_DETECTED:
      os << "linear dependency detected";
      return os;
    case jrl::qp::TerminationStatus::OVERCONSTRAINED_PROBLEM:
      os << "overconstrained problem";
      return os;
    default:
      os << "unknown failure (" << static_cast<int>(s) << ")";
      return os;
  }
}
} // namespace

namespace tvm
{

namespace solver
{
JRLQPLeastSquareSolver::JRLQPLeastSquareSolver(const JRLQPLSSolverOptions & options)
: LeastSquareSolver(options.verbose().value()), status_(jrl::qp::TerminationStatus::UNKNOWN), autoMinNorm_(false),
  big_number_(options.big_number().value()),
  damping_(options.damping().value())
{}

void JRLQPLeastSquareSolver::initializeBuild_(int nObj, int nEq, int nIneq, bool useBounds)
{
  resize_(nObj, nEq, nIneq, useBounds);

  autoMinNorm_ = false;
}

JRLQPLeastSquareSolver::ImpactFromChanges JRLQPLeastSquareSolver::resize_(int nObj,
                                                                                int nEq,
                                                                                int nIneq,
                                                                                bool useBounds)
{
  int n = variables().totalSize();
  int nCstr = nEq + nIneq;
  ImpactFromChanges impact;

  impact.objectives_ = ImpactFromChanges::willReallocate(D_, nObj, n);
  D_.resize(nObj, n);
  D_.setZero();
  e_.resize(nObj);
  Q_.resize(n, n);
  a_.resize(n);
  impact.equalityConstraints_ = ImpactFromChanges::willReallocate(C_, nCstr, n);
  C_.resize(nCstr, n);
  Ct_.resize(n, nCstr);
  C_.setZero();
  bl_ = Eigen::VectorXd::Constant(nCstr, -big_number_);
  bu_ = Eigen::VectorXd::Constant(nCstr, +big_number_);
  int nbnd = useBounds ? n : 0;
  impact.bounds_ = ImpactFromChanges::willReallocate(xl_, n);
  xl_ = Eigen::VectorXd::Constant(n, -big_number_);
  xu_ = Eigen::VectorXd::Constant(n, +big_number_);
  xStar_.resize(n);
 
  gi_.resize(n, nCstr, useBounds);

  impact.inequalityConstraints_ = impact.equalityConstraints_;
  return impact;
}

void JRLQPLeastSquareSolver::addBound_(LinearConstraintPtr bound, RangePtr range, bool first)
{
  scheme::internal::AssignmentTarget target(range, xl_, xu_);
  addAssignement(bound, target, bound->variables()[0], first);
}

void JRLQPLeastSquareSolver::addEqualityConstraint_(LinearConstraintPtr cstr)
{
  RangePtr r = std::make_shared<Range>(nextEqualityConstraintRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, C_, bl_, bu_, constraint::RHS::AS_GIVEN);
  addAssignement(cstr, nullptr, target, variables(), substitutions());
}

void JRLQPLeastSquareSolver::addIneqalityConstraint_(LinearConstraintPtr cstr)
{
  RangePtr r = std::make_shared<Range>(nextInequalityConstraintRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, C_, bl_, bu_, constraint::RHS::AS_GIVEN);
  addAssignement(cstr, nullptr, target, variables(), substitutions());
}

void JRLQPLeastSquareSolver::addObjective_(LinearConstraintPtr cstr,
                                              SolvingRequirementsPtr req,
                                              double additionalWeight)
{
  RangePtr r = std::make_shared<Range>(nextObjectiveRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, D_, e_, constraint::Type::EQUAL, constraint::RHS::OPPOSITE);
  addAssignement(cstr, req, target, variables(), substitutions(), additionalWeight);
}

void JRLQPLeastSquareSolver::setMinimumNorm_()
{
  autoMinNorm_ = true;
  Q_.setIdentity();
  a_.setZero();
}

void JRLQPLeastSquareSolver::resetBounds_()
{
  xl_.setConstant(-big_number_);
  xu_.setConstant(+big_number_);
}

void JRLQPLeastSquareSolver::preAssignmentProcess_()
{
  D_.setZero();
}

void JRLQPLeastSquareSolver::postAssignmentProcess_()
{
  if(autoMinNorm_)
  {
    // GoldfarbIdnaniSolver overwrites Q_, so we need to reset it
    Q_.setIdentity();
  }
  else
  {
    // a = D^T e
    a_.noalias() = D_.transpose() * e_;

    // Q = D^T D
    Q_.noalias() = D_.transpose() * D_; // TODO check if this can be optimized: JRLQP need only half the matrix.
    Q_.diagonal().array() += damping_;
  }

  // Transpose C as constraints in GoldfarbIdnaniSolver are of he form l<=C^T x<=u
  // TODO: change when assignment will be available for transposed target
  Ct_ = C_.transpose();
}

bool JRLQPLeastSquareSolver::solve_()
{
  status_ = gi_.solve(Q_, a_, Ct_, bl_, bu_, xl_, xu_);
  xStar_ = gi_.solution();
  return status_ == jrl::qp::TerminationStatus::SUCCESS;
}

const Eigen::VectorXd & JRLQPLeastSquareSolver::result_() const
{
  return xStar_;
}

Range JRLQPLeastSquareSolver::nextEqualityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const
{
  return {eqSize_ + ineqSize_, cstr.size()};
}

Range JRLQPLeastSquareSolver::nextInequalityConstraintRange_(
    const constraint::abstract::LinearConstraint & cstr) const
{
  return {eqSize_ + ineqSize_, constraintSize(cstr)};
}

Range JRLQPLeastSquareSolver::nextObjectiveRange_(const constraint::abstract::LinearConstraint & cstr) const
{
  return {objSize_, cstr.size()};
}

void JRLQPLeastSquareSolver::removeBounds_(const Range & range)
{
  xl_.segment(range.start, range.dim).setConstant(-big_number_);
  xu_.segment(range.start, range.dim).setConstant(+big_number_);
}

void JRLQPLeastSquareSolver::updateEqualityTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(C_, bl_, bu_);
}

void JRLQPLeastSquareSolver::updateInequalityTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(C_, bl_, bu_);
}

void JRLQPLeastSquareSolver::updateBoundTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(VectorRef(xl_), xu_);
}

void JRLQPLeastSquareSolver::updateObjectiveTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(MatrixRef(D_), e_);
}

void JRLQPLeastSquareSolver::applyImpactLogic(ImpactFromChanges & impact)
{
  if(impact.equalityConstraints_)
    impact.inequalityConstraints_ = true;
  if(impact.inequalityConstraints_)
    impact.equalityConstraints_ = true;
}

void JRLQPLeastSquareSolver::printProblemData_() const
{
  std::cout << "Q =\n" << Q_ << std::endl;
  std::cout << "a = " << a_.transpose() << std::endl;
  std::cout << "C =\n" << C_ << std::endl;
  std::cout << "bl = " << bl_.transpose() << std::endl;
  std::cout << "bu = " << bu_.transpose() << std::endl;
  std::cout << "xl = " << xl_.transpose() << std::endl;
  std::cout << "xu = " << xu_.transpose() << std::endl;
}

void JRLQPLeastSquareSolver::printDiagnostic_() const
{
  std::cout << "JRLQP status code = " << status_ << " (0 is success)" << std::endl;
}

std::unique_ptr<abstract::LSSolverFactory> JRLQPLSSolverFactory::clone() const
{
  return std::make_unique<JRLQPLSSolverFactory>(*this);
}

JRLQPLSSolverFactory::JRLQPLSSolverFactory(const JRLQPLSSolverOptions & options)
: LSSolverFactory("jrlqp"), options_(options)
{}

std::unique_ptr<abstract::LeastSquareSolver> JRLQPLSSolverFactory::createSolver() const
{
  return std::make_unique<JRLQPLeastSquareSolver>(options_);
}
} // namespace solver

} // namespace tvm