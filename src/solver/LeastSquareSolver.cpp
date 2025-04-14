/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/VariableVector.h>
#include <tvm/solver/abstract/LeastSquareSolver.h>

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
          LeastSquareSolver::AssignmentVector & observed,
          LeastSquareSolver::MapToAssignment & target)
  : observedSize_(observed.size()), observed_(observed), target_(target[cstr.get()])
  {}
  ~AutoMap()
  {
    for(size_t i = observedSize_; i < observed_.size(); ++i)
      target_.push_back(observed_[i].get());
  }

private:
  size_t observedSize_;
  LeastSquareSolver::AssignmentVector & observed_;
  LeastSquareSolver::AssignmentPtrVector & target_;
};
} // namespace

namespace tvm::solver::abstract
{
LeastSquareSolver::LeastSquareSolver(bool verbose)
: objSize_(-1), eqSize_(-1), ineqSize_(-1), buildInProgress_(false), verbose_(verbose), variables_(nullptr),
  subs_(nullptr)
{}

void LeastSquareSolver::startBuild(const VariableVector & x,
                                   int nObj,
                                   int nEq,
                                   int nIneq,
                                   bool useBounds,
                                   const hint::internal::Substitutions * const subs)
{
  assert(nObj >= 0);
  assert(nEq >= 0);
  assert(nIneq >= 0);

  buildInProgress_ = true;
  variables_ = &x;

  objectiveToAssignments_.clear();
  equalityConstraintToAssignments_.clear();
  inequalityConstraintToAssignments_.clear();
  boundToAssignments_.clear();
  assignments_.clear();

  subs_ = subs;

  initializeBuild_(nObj, nEq, nIneq, useBounds);
  nEq_ = nEq;
  nIneq_ = nIneq;
  nObj_ = nObj;
  objSize_ = 0;
  eqSize_ = 0;
  ineqSize_ = 0;
}

void LeastSquareSolver::finalizeBuild()
{
  assert(nObj_ == objSize_);
  assert(nEq_ == eqSize_);
  assert(nIneq_ == ineqSize_);
  buildInProgress_ = false;
}

void LeastSquareSolver::addBound(LinearConstraintPtr bound)
{
  assert(buildInProgress_ && "Attempting to add a bound without calling startBuild first");
  assert(bound->variables().numberOfVariables() == 1 && "A bound constraint can be only on one variable.");
  const auto & xi = bound->variables()[0];
  RangePtr range = std::make_shared<Range>(xi->getMappingIn(variables()));

  AutoMap autoMap(bound, assignments_, boundToAssignments_);
  addBound_(bound, range, false);
}

void LeastSquareSolver::addConstraint(LinearConstraintPtr cstr)
{
  assert(buildInProgress_ && "Attempting to add a constraint without calling startBuild first");

  if(cstr->isEquality())
  {
    AutoMap autoMap(cstr, assignments_, equalityConstraintToAssignments_);
    addEqualityConstraint_(cstr);
    eqSize_ += constraintSize(*cstr);
  }
  else
  {
    AutoMap autoMap(cstr, assignments_, inequalityConstraintToAssignments_);
    addIneqalityConstraint_(cstr);
    ineqSize_ += constraintSize(*cstr);
  }
}

void LeastSquareSolver::addObjective(LinearConstraintPtr obj, SolvingRequirementsPtr req, double additionalWeight)
{
  assert(req->priorityLevel().value() != 0);
  assert(buildInProgress_ && "Attempting to add an objective without calling startBuild first");

  if(req->violationEvaluation().value() != requirements::ViolationEvaluationType::L2)
  {
    throw std::runtime_error(
        "[LeastSquareSolver::addObjective]: least-squares only support L2 norm for violation evaluation");
  }
  AutoMap autoMap(obj, assignments_, objectiveToAssignments_);
  addObjective_(obj, req, additionalWeight);
  objSize_ += obj->size();
}

void LeastSquareSolver::setMinimumNorm()
{
  assert(nObj_ == variables().totalSize());
  if(!buildInProgress_)
  {
    throw std::runtime_error("[LeastSquareSolver]: attempting to add an objective without calling startBuild first");
  }
  setMinimumNorm_();
}

bool LeastSquareSolver::solve()
{
  if(buildInProgress_)
  {
    throw std::runtime_error("[LeastSquareSolver]: attempting to solve while in build mode");
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
      std::cout << "[LeastSquareSolver::solve] solution: " << result_().transpose() << std::endl;
    }
  }

  return b;
}

const Eigen::VectorXd & LeastSquareSolver::result() const { return result_(); }

int LeastSquareSolver::constraintSize(const constraint::abstract::LinearConstraint & c) const
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

void LeastSquareSolver::process(const internal::SolverEvents & se)
{
  updateWeights(se);
  auto impactRemove = processRemovedConstraints(se);
  bool needMappingUpdate = updateVariables(se);
  auto impactAdd = previewAddedConstraints(se);
  auto impactResize = resize_(nObj_, nEq_, nIneq_, boundToAssignments_.size() > 0 || se.addedBounds().size() > 0);

  if(needMappingUpdate)
  {
    for(auto & a : assignments_)
      a->assignment.onUpdatedMapping(variables(), false);
  }

  // Update the target ranges when needed.
  // We could be much more fine grain as added constraints could be placed last,
  // and not update existing constraints if only constraints are added but this
  // makes things much more complex.
  auto impact = impactRemove || impactAdd;
  auto updateTargetRange = [](const auto & map, int & cumSize, auto rangeProvider, auto sizeProvider) {
    for(const auto & ca : map)
    {
      Range r = rangeProvider(*ca.first);
      for(auto & a : ca.second)
        a->assignment.target().range().start = r.start;
      cumSize += sizeProvider(*ca.first);
    }
  };

  // We first reset all necessary size accumulators, as they can be used together for deciding ranges.
  if(impact.equalityConstraints_)
    eqSize_ = 0;
  if(impact.inequalityConstraints_)
    ineqSize_ = 0;
  if(impact.objectives_)
    objSize_ = 0;

  if(impact.equalityConstraints_)
    updateTargetRange(
        equalityConstraintToAssignments_, eqSize_, [&](const auto & c) { return nextEqualityConstraintRange_(c); },
        [&](const auto & c) { return constraintSize(c); });

  if(impact.inequalityConstraints_)
    updateTargetRange(
        inequalityConstraintToAssignments_, ineqSize_,
        [&](const auto & c) { return nextInequalityConstraintRange_(c); },
        [&](const auto & c) { return constraintSize(c); });

  int dummy = 0;
  if(impact.bounds_ || needMappingUpdate) // needMappingUpdate because a variable might have been added or removed
    updateTargetRange(
        boundToAssignments_, dummy, [&](const auto & b) { return b.variables()[0]->getMappingIn(variables()); },
        [](const auto &) { return 0; });

  if(impact.objectives_)
    updateTargetRange(
        objectiveToAssignments_, objSize_, [&](const auto & c) { return nextObjectiveRange_(c); },
        [&](const auto & c) { return c.size(); });

  // Update the matrices and vectors of the target when needed
  auto updateTargetData = [](const auto & map, auto updateFn) {
    for(const auto & ca : map)
    {
      for(auto & a : ca.second)
        updateFn(a->assignment.target());
    }
  };

  impact.orAssign(impactResize); // There are case where no resize are necessary (e.g. a constraint was removed and
  applyImpactLogic(impact);      // another added with same size) but the target need to change.
  if(impact.equalityConstraints_)
    updateTargetData(equalityConstraintToAssignments_, [&](auto & target) { updateEqualityTargetData(target); });

  if(impact.inequalityConstraints_)
    updateTargetData(inequalityConstraintToAssignments_, [&](auto & target) { updateInequalityTargetData(target); });

  if(impact.bounds_)
    updateTargetData(boundToAssignments_, [&](auto & target) { updateBoundTargetData(target); });

  if(impact.objectives_)
    updateTargetData(objectiveToAssignments_, [&](auto & target) { updateObjectiveTargetData(target); });

  // Final update of the impacted mappings
  if(needMappingUpdate)
  {
    for(auto & a : assignments_)
      a->assignment.onUpdatedTarget(); // onUpdateTarget also takes into account the mapping change due to variable.
  }
  else
  {
    auto updateMapping = [](const auto & map) {
      for(const auto & ca : map)
      {
        for(auto & a : ca.second)
          a->assignment.onUpdatedTarget();
      }
    };

    if(impact.equalityConstraints_)
      updateMapping(equalityConstraintToAssignments_);
    if(impact.inequalityConstraints_)
      updateMapping(inequalityConstraintToAssignments_);
    if(impact.bounds_)
      updateMapping(boundToAssignments_);
    if(impact.objectives_)
      updateMapping(objectiveToAssignments_);
  }

  processAddedConstraints(se);
}

void LeastSquareSolver::updateWeights(const internal::SolverEvents & se)
{
  const auto & we = se.weightEvents();

  for(const auto & e : we)
  {
    auto [c, p, scalar, vector] = e;

    // We might have change of weights on constraints that were not added yet (because process of
    // of weight arise before processing added constraints - if we'd process the weight after, we
    // could have change of weight on removed constraints). Therefore we need to ignore constraints
    // that were not found in the map.
    auto it = objectiveToAssignments_.find(c);
    if(it == objectiveToAssignments_.end())
      continue;
    auto & assignments = it->second;

    for(auto & a : assignments)
    {
      bool needReprocess = (scalar && !a->assignment.changeScalarWeightIsAllowed())
                           || (vector && !a->assignment.changeVectorWeightIsAllowed());
      if(needReprocess)
      {
        a->assignment = tvm::scheme::internal::Assignment::reprocess(a->assignment, variables(), substitutions());
      }
      else
      {
        a->assignment.onUpdateWeights(scalar, vector);
      }
    }
  }
}

bool LeastSquareSolver::updateVariables(const internal::SolverEvents & se)
{
  return (!(se.removedVariables().empty() && se.addedVariables().empty())) || se.hasHiddenVariableChange();
}

LeastSquareSolver::ImpactFromChanges LeastSquareSolver::processRemovedConstraints(const internal::SolverEvents & se)
{
  for(const auto & [c, _] : se.removedConstraints())
  {
    if(c->isEquality())
    {
      nEq_ -= constraintSize(*c);
      const auto & assignments = equalityConstraintToAssignments_[c.get()];
      for(auto & a : assignments)
        a->markedForRemoval = true;
      equalityConstraintToAssignments_.erase(c.get());
    }
    else
    {
      nIneq_ -= constraintSize(*c);
      const auto & assignments = inequalityConstraintToAssignments_[c.get()];
      for(auto & a : assignments)
        a->markedForRemoval = true;
      inequalityConstraintToAssignments_.erase(c.get());
    }
  }

  for(const auto & o : se.removedObjectives())
  {
    nObj_ -= o->size();
    const auto & assignments = objectiveToAssignments_[o.get()];
    for(auto & a : assignments)
      a->markedForRemoval = true;
    objectiveToAssignments_.erase(o.get());
  }

  for(const auto & b : se.removedBounds())
  {
    const auto & assignments = boundToAssignments_[b.get()];
    for(auto & a : assignments)
      a->markedForRemoval = true;
    boundToAssignments_.erase(b.get());
  }

  // clear assignments
  auto it =
      std::remove_if(assignments_.begin(), assignments_.end(), [](const auto & it) { return it->markedForRemoval; });
  assignments_.erase(it, assignments_.end());

  ImpactFromChanges impact = {!se.removedConstraints().empty(), !se.removedConstraints().empty(),
                              !se.removedBounds().empty(), !se.removedObjectives().empty()};
  applyImpactLogic(impact);
  return impact;
}

LeastSquareSolver::ImpactFromChanges LeastSquareSolver::previewAddedConstraints(const internal::SolverEvents & se)
{
  // When used as part of process, the following code will generate target ranges
  // outside of the matrices bounds. It relies on the fact that the mapping of the
  // assignment is updated after.
  eqSize_ = nEq_;
  ineqSize_ = nIneq_;
  objSize_ = nObj_;
  for(const auto & [c, _] : se.addedConstraints())
  {
    if(c->isEquality())
      nEq_ += constraintSize(*c);
    else
      nIneq_ += constraintSize(*c);
  }

  for(const auto & o : se.addedObjectives())
    nObj_ += o.c->size();

  ImpactFromChanges impact = {nEq_ != eqSize_, nIneq_ != ineqSize_, false, !se.addedObjectives().empty()};
  applyImpactLogic(impact);
  return impact;
}

void LeastSquareSolver::processAddedConstraints(const internal::SolverEvents & se)
{
  buildInProgress_ = true;
  for(const auto & [c, _] : se.addedConstraints())
    addConstraint(c);

  for(const auto & b : se.addedBounds())
    addBound(b);

  for(const auto & o : se.addedObjectives())
    addObjective(o.c, o.req, o.scalarizationWeight);

  assert(nObj_ == objSize_);
  assert(nEq_ == eqSize_);
  assert(nIneq_ == ineqSize_);
  buildInProgress_ = false;
}

LeastSquareSolver::ImpactFromChanges LeastSquareSolver::ImpactFromChanges::operator||(bool b)
{
  return {this->equalityConstraints_ || b, this->inequalityConstraints_ || b, this->bounds_ || b,
          this->objectives_ || b};
}

LeastSquareSolver::ImpactFromChanges LeastSquareSolver::ImpactFromChanges::operator||(const ImpactFromChanges & other)
{
  return {this->equalityConstraints_ || other.equalityConstraints_,
          this->inequalityConstraints_ || other.inequalityConstraints_, this->bounds_ || other.bounds_,
          this->objectives_ || other.objectives_};
}

LeastSquareSolver::ImpactFromChanges & LeastSquareSolver::ImpactFromChanges::orAssign(const ImpactFromChanges & other)
{
  *this = *this || other;
  return *this;
}

void LeastSquareSolver::applyImpactLogic(ImpactFromChanges &)
{
  // do nothing;
}
} // namespace tvm::solver::abstract
