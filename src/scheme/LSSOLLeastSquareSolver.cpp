#include <tvm/scheme/LSSOLLeastSquareSolver.h>

#include <tvm/scheme/internal/AssignmentTarget.h>

namespace tvm
{

namespace scheme
{
  LSSOLLeastSquareSolver::LSSOLLeastSquareSolver(double big_number)
    : LeastSquareSolver()
    , cl_(l_.tail(0))
    , cu_(u_.tail(0))
    , big_number_(big_number)
  {
  }

  void LSSOLLeastSquareSolver::initializeBuild_(int m1, int me, int mi, bool useBounds)
  {
    int n = variables().totalSize();
    int m0 = me + mi;
    A_.resize(m1, n);
    A_.setZero();
    C_.resize(m0, n);
    C_.setZero();
    b_.resize(m1);
    b_.setZero();
    l_ = Eigen::VectorXd::Constant(m0 + n, -big_number_);
    u_ = Eigen::VectorXd::Constant(m0 + n, +big_number_);
    cl_ = l_.tail(m0);
    cu_ = u_.tail(m0);
    ls_.resize(n, m0, Eigen::lssol::eType::LS1);
  }

  void LSSOLLeastSquareSolver::addBound_(LinearConstraintPtr bound, RangePtr range, bool first)
  {
    internal::AssignmentTarget target(range, l_, u_);
    addAssignement(bound, target, bound->variables()[0], first);
  }

  void LSSOLLeastSquareSolver::addEqualityConstraint_(LinearConstraintPtr cstr)
  {
    RangePtr r = std::make_shared<Range>(eqSize_+ineqSize_, cstr->size());
    internal::AssignmentTarget target(r, C_, cl_, cu_, constraint::RHS::AS_GIVEN);
    addAssignement(cstr, nullptr, target, variables(), *substitutions());
  }

  void LSSOLLeastSquareSolver::addIneqalityConstraint_(LinearConstraintPtr cstr)
  {
    RangePtr r = std::make_shared<Range>(eqSize_ + ineqSize_, cstr->size());
    internal::AssignmentTarget target(r, C_, cl_, cu_, constraint::RHS::AS_GIVEN);
    addAssignement(cstr, nullptr, target, variables(), *substitutions());
  }

  void LSSOLLeastSquareSolver::addObjective_(LinearConstraintPtr cstr, SolvingRequirementsPtr req, double additionalWeight)
  {
    RangePtr r = std::make_shared<Range>(objSize_, cstr->size());
    internal::AssignmentTarget target(r, A_, b_, constraint::Type::EQUAL, constraint::RHS::AS_GIVEN);
    addAssignement(cstr, req, target, variables(), *substitutions(), additionalWeight);
  }

  bool LSSOLLeastSquareSolver::solve_()
  {
    return ls_.solve(A_, b_, C_, l_, u_);
  }


  LSSOLLeastSquareSolverConfiguration::LSSOLLeastSquareSolverConfiguration(double big_number)
    : LeastSquareSolverConfiguration("lssol")
    , big_number_(big_number)
  {
  }
  
  std::unique_ptr<abstract::LeastSquareSolver> LSSOLLeastSquareSolverConfiguration::createSolver() const
  {
    return std::make_unique<LSSOLLeastSquareSolver>(big_number_);
  }
}

}