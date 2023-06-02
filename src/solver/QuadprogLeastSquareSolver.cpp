/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/solver/QuadprogLeastSquareSolver.h>

#include <tvm/scheme/internal/AssignmentTarget.h>

#include <iostream>

namespace tvm
{

namespace solver
{
QuadprogLeastSquareSolver::QuadprogLeastSquareSolver(const QuadprogLSSolverOptions & options)
: LeastSquareSolver(options.verbose().value()), Aineq_(A_.middleRows(0, 0)), bineq_(b_.segment(0, 0)),
  xl_(b_.segment(0, 0)), xu_(b_.segment(0, 0)), autoMinNorm_(false), big_number_(options.big_number().value()),
  damping_(options.damping().value()), cholesky_(options.cholesky().value()),
  choleskyDamping_(options.choleskyDamping().value())
{}

void QuadprogLeastSquareSolver::initializeBuild_(int nObj, int nEq, int nIneq, bool useBounds)
{
  resize_(nObj, nEq, nIneq, useBounds);

  autoMinNorm_ = false;
}

QuadprogLeastSquareSolver::ImpactFromChanges QuadprogLeastSquareSolver::resize_(int nObj,
                                                                                int nEq,
                                                                                int nIneq,
                                                                                bool useBounds)
{
  int n = variables().totalSize();
  int nCstr = nEq + nIneq;
  underspecifiedObj_ = nObj < n;
  ImpactFromChanges impact;

  if(cholesky_ && underspecifiedObj_)
  {
    impact.objectives_ = ImpactFromChanges::willReallocate(D_, nObj + n, n);
    D_.resize(nObj + n, n);
    D_.setZero();
    D_.bottomRows(n).diagonal().setConstant(choleskyDamping_);
  }
  else
  {
    impact.objectives_ = ImpactFromChanges::willReallocate(D_, nObj, n);
    D_.resize(nObj, n);
    D_.setZero();
  }
  e_.resize(nObj);
  if(!cholesky_)
    Q_.resize(n, n);
  c_.resize(n);
  if(useBounds)
  {
    //               | A_cstr |
    // A needs to be |   -I   | with A_cstr = |  A_eq  |
    //               |   I    |               | A_ineq |
    impact.equalityConstraints_ = ImpactFromChanges::willReallocate(A_, nCstr + 2 * n, n);
    nIneqInclBounds_ = nIneq + 2 * n;
    A_.resize(nCstr + 2 * n, n);
    A_.setZero();
    b_.resize(nCstr + 2 * n);
    A_.middleRows(nCstr, n).diagonal().setConstant(-1);
    A_.bottomRows(n).diagonal().setConstant(1);
    new(&xl_) VectorXdSeg(b_.segment(nCstr, n));
    new(&xu_) VectorXdSeg(b_.segment(nCstr + n, n));
    xl_.setConstant(-big_number_);
    xu_.setConstant(+big_number_);
  }
  else
  {
    impact.equalityConstraints_ = ImpactFromChanges::willReallocate(A_, nCstr, n);
    nIneqInclBounds_ = nIneq;
    A_.resize(nCstr, n);
    A_.setZero();
    b_.resize(nCstr);
    new(&xl_) VectorXdSeg(b_.segment(nCstr, 0));
    new(&xu_) VectorXdSeg(b_.segment(nCstr, 0));
  }
  new(&Aineq_) MatrixXdRows(A_.middleRows(nEq, nIneq));
  new(&bineq_) VectorXdSeg(b_.segment(nEq, nIneq));
  if(useBounds)
    qpd_.problem(n, nEq, nIneq + 2 * n);
  else
    qpd_.problem(n, nEq, nIneq);
  if(cholesky_)
  {
    if(underspecifiedObj_)
      new(&qr_) Eigen::HouseholderQR<Eigen::MatrixXd>(nObj + n, n);
    else
      new(&qr_) Eigen::HouseholderQR<Eigen::MatrixXd>(nObj, n);
  }

  impact.inequalityConstraints_ = impact.equalityConstraints_;
  impact.bounds_ = impact.equalityConstraints_;
  return impact;
}

void QuadprogLeastSquareSolver::addBound_(LinearConstraintPtr bound, RangePtr range, bool first)
{
  // We would like to write the lower bound as -x <= -xl, which means that xl_ should be
  // filled with -xl, but there is currently no path in Assignment to do that properly.
  // Instead, we fill xl_ as if we wanted xl <= x <= xu, and change the sign of xl_ in
  // postAssignmentProcess_(). The -x comes from the corresponding rows of A being set
  // to -I in initializeBuild_
  // TODO: extend Assignment for that.
  scheme::internal::AssignmentTarget target(range, xl_, xu_);
  addAssignement(bound, target, bound->variables()[0], first);
}

void QuadprogLeastSquareSolver::addEqualityConstraint_(LinearConstraintPtr cstr)
{
  RangePtr r = std::make_shared<Range>(nextEqualityConstraintRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, A_, b_, constraint::Type::EQUAL, constraint::RHS::AS_GIVEN);
  addAssignement(cstr, nullptr, target, variables(), substitutions());
}

void QuadprogLeastSquareSolver::addIneqalityConstraint_(LinearConstraintPtr cstr)
{
  RangePtr r = std::make_shared<Range>(nextInequalityConstraintRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, Aineq_, bineq_, constraint::Type::LOWER_THAN, constraint::RHS::AS_GIVEN);
  addAssignement(cstr, nullptr, target, variables(), substitutions());
}

void QuadprogLeastSquareSolver::addObjective_(LinearConstraintPtr cstr,
                                              SolvingRequirementsPtr req,
                                              double additionalWeight)
{
  RangePtr r = std::make_shared<Range>(nextObjectiveRange_(*cstr));
  scheme::internal::AssignmentTarget target(r, D_, e_, constraint::Type::EQUAL, constraint::RHS::OPPOSITE);
  addAssignement(cstr, req, target, variables(), substitutions(), additionalWeight);
}

void QuadprogLeastSquareSolver::setMinimumNorm_()
{
  autoMinNorm_ = true;
  Q_.setIdentity();
  c_.setZero();
}

void QuadprogLeastSquareSolver::resetBounds_()
{
  xl_.setConstant(-big_number_);
  xu_.setConstant(+big_number_);
}

void QuadprogLeastSquareSolver::preAssignmentProcess_()
{
  // Some variables may be unbounded, which means no assignment will set the bounds to
  // the correct value. Since the signs on xl will be flipped later, we need to reset this
  // correct value.
  xl_.setConstant(-big_number_);

  // In some instance, D is overwritten, we need to reset it.
  if(!autoMinNorm_ && cholesky_)
  {
    D_.setZero();
    if(underspecifiedObj_)
    {
      int n = variables().totalSize();
      D_.bottomRows(n).diagonal().setConstant(choleskyDamping_);
    }
  }
}

void QuadprogLeastSquareSolver::postAssignmentProcess_()
{
  // we need to flip the signs for xl
  xl_ = -xl_;

  // Quadprog does not solve least-square cost, but quadratic cost.
  // We then either need to form the quadratic cost, or to get the Cholesky
  // decomposition of this quadratic cost.
  if(!autoMinNorm_)
  {
    // c = D^T e
    c_.noalias() = D_.topRows(nObj_).transpose() * e_;

    if(cholesky_)
    {
      // The cholesky decomposition of Q = D^T D is given by the R factor of
      // the QR decomposition of D: if D = U R with U orthogonal, D^T D = R^T R.
      int n = variables().totalSize();
      qr_.compute(D_);
      // we put R^{-1} in D
      D_.topRows(n).setIdentity();
      qr_.matrixQR().topRows(n).template triangularView<Eigen::Upper>().solveInPlace(D_.topRows(n));
    }
    else
    {
      // Q = D^T D
      Q_.noalias() =
          D_.transpose() * D_; // TODO check if this can be optimized: Quadprog might need only half the matrix
      Q_.diagonal().array() += damping_;
    }
  }
}

bool QuadprogLeastSquareSolver::solve_()
{
  if(cholesky_ && !autoMinNorm_)
  {
    int n = variables().totalSize();
    return qpd_.solve(D_.topRows(n), c_, A_.topRows(nEq_), b_.topRows(nEq_), A_.bottomRows(nIneqInclBounds_),
                      b_.bottomRows(nIneqInclBounds_), true);
  }
  else
  {
    return qpd_.solve(Q_, c_, A_.topRows(nEq_), b_.topRows(nEq_), A_.bottomRows(nIneqInclBounds_),
                      b_.bottomRows(nIneqInclBounds_), false);
  }
}

const Eigen::VectorXd & QuadprogLeastSquareSolver::result_() const { return qpd_.result(); }

Range QuadprogLeastSquareSolver::nextEqualityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const
{
  return {eqSize_, cstr.size()};
}

Range QuadprogLeastSquareSolver::nextInequalityConstraintRange_(
    const constraint::abstract::LinearConstraint & cstr) const
{
  return {ineqSize_, constraintSize(cstr)};
}

Range QuadprogLeastSquareSolver::nextObjectiveRange_(const constraint::abstract::LinearConstraint & cstr) const
{
  return {objSize_, cstr.size()};
}

void QuadprogLeastSquareSolver::removeBounds_(const Range & range)
{
  xl_.segment(range.start, range.dim).setConstant(-big_number_);
  xu_.segment(range.start, range.dim).setConstant(+big_number_);
}

void QuadprogLeastSquareSolver::updateEqualityTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(MatrixRef(A_), b_);
}

void QuadprogLeastSquareSolver::updateInequalityTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(MatrixRef(Aineq_), bineq_);
}

void QuadprogLeastSquareSolver::updateBoundTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(VectorRef(xl_), xu_);
}

void QuadprogLeastSquareSolver::updateObjectiveTargetData(scheme::internal::AssignmentTarget & target)
{
  target.changeData(MatrixRef(D_), e_);
}

void QuadprogLeastSquareSolver::applyImpactLogic(ImpactFromChanges & impact)
{
  if(impact.equalityConstraints_)
    impact.inequalityConstraints_ = true;
  if(impact.inequalityConstraints_)
    impact.bounds_ = true;
}

void QuadprogLeastSquareSolver::printProblemData_() const
{
  if(cholesky_)
  {
    int n = variables().totalSize();
    std::cout << "R =\n"
              << qr_.matrixQR().topRows(n).template triangularView<Eigen::Upper>().toDenseMatrix() << std::endl;
  }
  else
    std::cout << "Q =\n" << Q_ << std::endl;
  std::cout << "c = " << c_.transpose() << std::endl;
  std::cout << "A =\n" << A_ << std::endl;
  std::cout << "b = " << b_.transpose() << std::endl;
}

void QuadprogLeastSquareSolver::printDiagnostic_() const
{
  std::cout << "Quadprog fail code = " << qpd_.fail() << " (0 is success)" << std::endl;
}

std::unique_ptr<abstract::LSSolverFactory> QuadprogLSSolverFactory::clone() const
{
  return std::make_unique<QuadprogLSSolverFactory>(*this);
}

QuadprogLSSolverFactory::QuadprogLSSolverFactory(const QuadprogLSSolverOptions & options)
: LSSolverFactory("quadprog"), options_(options)
{}

std::unique_ptr<abstract::LeastSquareSolver> QuadprogLSSolverFactory::createSolver() const
{
  return std::make_unique<QuadprogLeastSquareSolver>(options_);
}
} // namespace solver

} // namespace tvm
