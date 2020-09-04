/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/solver/QLDLeastSquareSolver.h>

#include <tvm/scheme/internal/AssignmentTarget.h>

#include <iostream>

namespace tvm
{

namespace solver
{
  QLDLeastSquareSolver::QLDLeastSquareSolver(const QLDLSSolverOptions& options)
    : LeastSquareSolver(options.verbose().value())
    , Aineq_(A_.bottomRows(0))
    , bineq_(b_.tail(0))
    , big_number_(options.big_number().value())
    , cholesky_(options.cholesky().value())
    , choleskyDamping_(options.choleskyDamping().value())
    , eps_(options.eps().value())
    , autoMinNorm_(false)
  {
  }

  void QLDLeastSquareSolver::initializeBuild_(int nObj, int nEq, int nIneq, bool)
  {
    resize_(nObj, nEq, nIneq, true);

    autoMinNorm_ = false;
  }

  QLDLeastSquareSolver::ImpactFromChanges QLDLeastSquareSolver::resize_(int nObj, int nEq, int nIneq, bool)
  {
    int n = variables().totalSize();
    int nCstr = nEq + nIneq;
    underspecifiedObj_ = nObj < n;
    ImpactFromChanges impact;

    if (cholesky_ && underspecifiedObj_)
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
    if (!cholesky_)
      Q_.resize(n, n);
    c_.resize(n);
    impact.equalityConstraints_ = ImpactFromChanges::willReallocate(A_, nCstr, n);
    A_.resize(nCstr, n);
    A_.setZero();
    b_.resize(nCstr);
    impact.bounds_ = ImpactFromChanges::willReallocate(xl_, n);
    xl_ = Eigen::VectorXd::Constant(n, -big_number_);
    xu_ = Eigen::VectorXd::Constant(n, +big_number_);
    new(&Aineq_) MatrixXdBottom(A_.bottomRows(nIneq));
    new(&bineq_) VectorXdTail(b_.tail(nIneq));
    if (underspecifiedObj_)
      qld_.problem(n, nEq, nIneq, n + nObj);
    else
      qld_.problem(n, nEq, nIneq, cholesky_?nObj:n);
    if (cholesky_)
    {
      if (underspecifiedObj_)
        new(&qr_) Eigen::HouseholderQR<Eigen::MatrixXd>(nObj + n, n);
      else
        new(&qr_) Eigen::HouseholderQR<Eigen::MatrixXd>(nObj, n);
    }

    impact.inequalityConstraints_ = impact.equalityConstraints_;
    return impact;
  }

  void QLDLeastSquareSolver::addBound_(LinearConstraintPtr bound, RangePtr range, bool first)
  {
    scheme::internal::AssignmentTarget target(range, xl_, xu_);
    addAssignement(bound, target, bound->variables()[0], first);
  }

  void QLDLeastSquareSolver::addEqualityConstraint_(LinearConstraintPtr cstr)
  {
    RangePtr r = std::make_shared<Range>(nextEqualityConstraintRange_(*cstr));
    scheme::internal::AssignmentTarget target(r, A_, b_, constraint::Type::EQUAL, constraint::RHS::OPPOSITE);
    addAssignement(cstr, nullptr, target, variables(), substitutions());
  }

  void QLDLeastSquareSolver::addIneqalityConstraint_(LinearConstraintPtr cstr)
  {
    RangePtr r = std::make_shared<Range>(nextInequalityConstraintRange_(*cstr));
    scheme::internal::AssignmentTarget target(r, Aineq_, bineq_, constraint::Type::GREATER_THAN, constraint::RHS::OPPOSITE);
    addAssignement(cstr, nullptr, target, variables(), substitutions());
  }

  void QLDLeastSquareSolver::addObjective_(LinearConstraintPtr cstr, SolvingRequirementsPtr req, double additionalWeight)
  {
    RangePtr r = std::make_shared<Range>(nextObjectiveRange_(*cstr));
    scheme::internal::AssignmentTarget target(r, D_, e_, constraint::Type::EQUAL, constraint::RHS::OPPOSITE);
    addAssignement(cstr, req, target, variables(), substitutions(), additionalWeight);
  }

  void QLDLeastSquareSolver::setMinimumNorm_()
  {
    autoMinNorm_ = true;
    Q_.setIdentity();
    c_.setZero();
  }

  void QLDLeastSquareSolver::preAssignmentProcess_()
  {
  }

  void QLDLeastSquareSolver::postAssignmentProcess_()
  {
    // QLD does not solve least-square cost, but quadratic cost.
    // We then either need to form the quadratic cost, or to get the Cholesky
    // decomposition of this quadratic cost.
    if (!autoMinNorm_)
    {
      // c = D^T e
      c_.noalias() = D_.topRows(nObj_).transpose() * e_;

      if (cholesky_)
      {
        // The cholesky decomposition of Q = D^T D is given by the R factor of
        // the QR decomposition of D: if D = U R with U orthogonal, D^T D = R^T R.
        qr_.compute(D_);
      }
      else
      {
        // Q = D^T D
        Q_.noalias() = D_.transpose() * D_;   //TODO check if this can be optimized: QLD might need only half the matrix
      }
    }
  }

  bool QLDLeastSquareSolver::solve_()
  {
    if (cholesky_ && !autoMinNorm_)
    {
      int n = variables().totalSize();
      return qld_.solve(qr_.matrixQR().topRows(n), c_,
                        A_, b_,
                        xl_, xu_,
                        nEq_,
                        true, eps_);
    }
    else
    {
      return qld_.solve(Q_, c_,
                        A_, b_,
                        xl_, xu_,
                        nEq_,
                        false, eps_);
    }
  }

  const Eigen::VectorXd& QLDLeastSquareSolver::result_() const
  {
    return qld_.result();
  }

  Range QLDLeastSquareSolver::nextEqualityConstraintRange_(const constraint::abstract::LinearConstraint& cstr) const
  {
    return { eqSize_, cstr.size() };
  }

  Range QLDLeastSquareSolver::nextInequalityConstraintRange_(const constraint::abstract::LinearConstraint& cstr) const
  {
    return { ineqSize_, constraintSize(cstr) };
  }

  Range QLDLeastSquareSolver::nextObjectiveRange_(const constraint::abstract::LinearConstraint& cstr) const
  {
    return { objSize_, cstr.size() };
  }

  void QLDLeastSquareSolver::removeBounds_(const Range& range)
  {
    xl_.segment(range.start, range.dim).setConstant(-big_number_);
    xu_.segment(range.start, range.dim).setConstant(+big_number_);
  }

  void QLDLeastSquareSolver::updateEqualityTargetData(scheme::internal::AssignmentTarget& target)
  {
    target.changeData(A_, b_);
  }

  void QLDLeastSquareSolver::updateInequalityTargetData(scheme::internal::AssignmentTarget& target)
  {
    target.changeData(Aineq_, bineq_);
  }

  void QLDLeastSquareSolver::updateBoundTargetData(scheme::internal::AssignmentTarget& target)
  {
    target.changeData(VectorRef(xl_), xu_);
  }

  void QLDLeastSquareSolver::updateObjectiveTargetData(scheme::internal::AssignmentTarget& target)
  {
    target.changeData(D_, e_);
  }

  void QLDLeastSquareSolver::applyImpactLogic(ImpactFromChanges& impact)
  {
    if (impact.equalityConstraints_) impact.inequalityConstraints_ = true;
  }

  void QLDLeastSquareSolver::printProblemData_() const
  {
    if (cholesky_)
    {
      int n = variables().totalSize();
      std::cout << "R =\n" << qr_.matrixQR().topRows(n).template triangularView<Eigen::Upper>().toDenseMatrix() << std::endl;
    }
    else
      std::cout << "Q =\n" << Q_ << std::endl;
    std::cout << "c = " << c_.transpose() << std::endl;
    std::cout << "A =\n" << A_ << std::endl;
    std::cout << "b = " << b_.transpose() << std::endl;
    std::cout << "xl = " << xl_.transpose() << std::endl;
    std::cout << "xu = " << xu_.transpose() << std::endl;
  }

  void QLDLeastSquareSolver::printDiagnostic_() const
  {
    std::cout << "QLD fail code = " << qld_.fail() << " (0 is success)" << std::endl;
  }

  std::unique_ptr<abstract::LSSolverFactory> QLDLSSolverFactory::clone() const
  {
    return std::make_unique<QLDLSSolverFactory>(*this);
  }

  QLDLSSolverFactory::QLDLSSolverFactory(const QLDLSSolverOptions& options)
    : LSSolverFactory("qld")
    , options_(options)
  {
  }

  std::unique_ptr<abstract::LeastSquareSolver> QLDLSSolverFactory::createSolver() const
  {
    return std::make_unique<QLDLeastSquareSolver>(options_);
  }

}

}
