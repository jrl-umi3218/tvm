/* Copyright 2022 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/VariableVector.h>
#include <tvm/solver/abstract/HierarchicalLeastSquareSolver.h>

#include <iostream>

namespace
{
using namespace tvm;
using namespace tvm::solver::abstract;
/** Helper class relying on RAII to update automatically the map from a constraint to
 * the corresponding assignments
 */
class AutoMap
{
public:
  /** Upon destruction of this object, pointers to all new elements in \a observed since
   * calling this constructor will be added to \a target[cstr.get()].
   */
  AutoMap(LinearConstraintPtr cstr,
          HierarchicalLeastSquareSolver::AssignmentVector & observed,
          HierarchicalLeastSquareSolver::MapToAssignment & target)
  : observedSize_(observed.size()), observed_(observed), target_(target[cstr.get()])
  {}
  ~AutoMap()
  {
    for(size_t i = observedSize_; i < observed_.size(); ++i)
      target_.push_back(observed_[i].get());
  }

private:
  size_t observedSize_;
  HierarchicalLeastSquareSolver::AssignmentVector & observed_;
  HierarchicalLeastSquareSolver::AssignmentPtrVector & target_;
};
} // namespace

namespace tvm::solver::abstract
{
HierarchicalLeastSquareSolver::HierarchicalLeastSquareSolver(bool verbose)
: eqSize_(), ineqSize_(), buildInProgress_(false), verbose_(verbose), variables_(nullptr), subs_(nullptr)
{}

void HierarchicalLeastSquareSolver::startBuild(const VariableVector & x,
                                               const std::vector<int> & nEq,
                                               const std::vector<int> & nIneq,
                                               bool useBounds,
                                               const hint::internal::Substitutions * const subs)
{
  assert(std::all_of(nEq.begin(), nEq.end(), [](int i) { return i >= 0; })
         && "Needs to have positive size for each level.");
  assert(std::all_of(nIneq.begin(), nIneq.end(), [](int i) { return i >= 0; })
         && "Needs to have positive size for each level.");
  assert(nEq.size() == nIneq.size());

  buildInProgress_ = true;
  variables_ = &x;

  auto nLvl = nEq.size();

  equalityConstraintToAssignments_.clear();
  equalityConstraintToAssignments_.resize(nLvl);
  inequalityConstraintToAssignments_.clear();
  inequalityConstraintToAssignments_.resize(nLvl);
  boundToAssignments_.clear();
  assignments_.clear();

  subs_ = subs;

  initializeBuild_(nEq, nIneq, useBounds);
  nEq_ = nEq;
  nIneq_ = nIneq;
  eqSize_.resize(nEq_.size(), 0);
  ineqSize_.resize(nEq_.size(), 0);
  useBounds_ = useBounds;
}

void HierarchicalLeastSquareSolver::finalizeBuild()
{
  assert(nEq_ == eqSize_);
  assert(nIneq_ == ineqSize_);
  buildInProgress_ = false;
}

void HierarchicalLeastSquareSolver::addBound(LinearConstraintPtr bound)
{
  assert(buildInProgress_ && "Attempting to add a bound without calling startBuild first");
  assert(bound->variables().numberOfVariables() == 1 && "A bound constraint can be only on one variable.");
  const auto & xi = bound->variables()[0];
  RangePtr range = std::make_shared<Range>(xi->getMappingIn(variables()));

  AutoMap autoMap(bound, assignments_, boundToAssignments_);
  addBound_(bound, range, false);
}

void HierarchicalLeastSquareSolver::addConstraint(LinearConstraintPtr cstr, SolvingRequirementsPtr req)
{
  int lvl = req->priorityLevel().value();

  assert(buildInProgress_ && "Attempting to add a constraint without calling startBuild first");
  assert(lvl >= 0 && lvl < nEq_.size());

  if(cstr->isEquality())
  {
    AutoMap autoMap(cstr, assignments_, equalityConstraintToAssignments_[lvl]);
    addEqualityConstraint_(cstr, req);
    eqSize_[lvl] += constraintSize(*cstr);
  }
  else
  {
    AutoMap autoMap(cstr, assignments_, inequalityConstraintToAssignments_[lvl]);
    addIneqalityConstraint_(cstr, req);
    ineqSize_[lvl] += constraintSize(*cstr);
  }
}

void HierarchicalLeastSquareSolver::setMinimumNorm()
{
  assert(nEq_.back() == variables().totalSize() && nIneq_.back() == 0);
  if(!buildInProgress_)
  {
    throw std::runtime_error(
        "[HierarchicalLeastSquareSolver]: attempting to add an objective without calling startBuild first");
  }
  setMinimumNorm_();
  eqSize_.back() = variables_->totalSize();
}

bool HierarchicalLeastSquareSolver::solve()
{
  if(buildInProgress_)
  {
    throw std::runtime_error("[HierarchicalLeastSquareSolver]: attempting to solve while in build mode");
  }

  resetBounds_();
  preAssignmentProcess_();
  for(auto & a : assignments_)
    a->assignment.run();
  postAssignmentProcess_();

  if(verbose_)
    printProblemData_();

  bool b = solve_();

  if(verbose_ || !b)
  {
    printDiagnostic_();
    if(verbose_)
    {
      std::cout << "[HierarchicalLeastSquareSolver::solve] solution: " << result_().transpose() << std::endl;
    }
  }

  return b;
}

const Eigen::VectorXd & HierarchicalLeastSquareSolver::result() const { return result_(); }

int HierarchicalLeastSquareSolver::constraintSize(const constraint::abstract::LinearConstraint & c) const
{
  if(c.type() != constraint::Type::DOUBLE_SIDED || handleDoubleSidedConstraint_())
  {
    return c.size();
  }
  else
  {
    return 2 * c.size();
  }
}

void HierarchicalLeastSquareSolver::process(const internal::SolverEvents & se)
{
  throw std::runtime_error("[HierarchicalLeastSquareSolver::process] Not implemented yet");
}

void HierarchicalLeastSquareSolver::updateWeights(const internal::SolverEvents & se)
{
  throw std::runtime_error("[HierarchicalLeastSquareSolver::updateWeights] Not implemented yet");
}

bool HierarchicalLeastSquareSolver::updateVariables(const internal::SolverEvents & se)
{
  return (!se.removedVariables().empty() || !se.addedVariables().empty()) || se.hasHiddenVariableChange();
}

HierarchicalLeastSquareSolver::ImpactFromChanges HierarchicalLeastSquareSolver::processRemovedConstraints(
    const internal::SolverEvents & se)
{
  throw std::runtime_error("[HierarchicalLeastSquareSolver::processRemovedConstraints] Not implemented yet");
  return {0};
}

HierarchicalLeastSquareSolver::ImpactFromChanges HierarchicalLeastSquareSolver::previewAddedConstraints(
    const internal::SolverEvents & se)
{
  throw std::runtime_error("[HierarchicalLeastSquareSolver::previewAddedConstraints] Not implemented yet");
  return {0};
}

void HierarchicalLeastSquareSolver::processAddedConstraints(const internal::SolverEvents & se)
{
  throw std::runtime_error("[HierarchicalLeastSquareSolver::processAddedConstraints] Not implemented yet");
}

HierarchicalLeastSquareSolver::ImpactFromChanges::ImpactFromChanges(int nLvl)
: equalityConstraints_(nLvl, false), inequalityConstraints_(nLvl, false)
{}

HierarchicalLeastSquareSolver::ImpactFromChanges::ImpactFromChanges(const std::vector<bool> & eq,
                                                                    std::vector<bool> & ineq,
                                                                    bool bounds)
: equalityConstraints_(eq), inequalityConstraints_(ineq), bounds_(bounds)
{}

HierarchicalLeastSquareSolver::ImpactFromChanges HierarchicalLeastSquareSolver::ImpactFromChanges::operator||(
    const ImpactFromChanges & other)
{
  assert(equalityConstraints_.size() == other.equalityConstraints_.size());
  ImpactFromChanges ret(static_cast<int>(equalityConstraints_.size()));
  std::transform(equalityConstraints_.begin(), equalityConstraints_.end(), other.equalityConstraints_.begin(),
                 ret.equalityConstraints_.begin(), [](bool b1, bool b2) { return b1 || b2; });
  std::transform(inequalityConstraints_.begin(), inequalityConstraints_.end(), other.inequalityConstraints_.begin(),
                 ret.inequalityConstraints_.begin(), [](bool b1, bool b2) { return b1 || b2; });
  ret.bounds_ = bounds_ || other.bounds_;
  return ret;
}

HierarchicalLeastSquareSolver::ImpactFromChanges & HierarchicalLeastSquareSolver::ImpactFromChanges::orAssign(
    const ImpactFromChanges & other)
{
  *this = *this || other;
  return *this;
}

void HierarchicalLeastSquareSolver::applyImpactLogic(ImpactFromChanges &)
{
  // do nothing;
}
} // namespace tvm::solver::abstract
