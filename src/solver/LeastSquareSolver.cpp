/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/solver/abstract/LeastSquareSolver.h>
#include <tvm/VariableVector.h>

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
    AutoMap(LinearConstraintPtr cstr, LeastSquareSolver::AssignmentVector& observed, LeastSquareSolver::MapToAssignment& target)
      : observedSize_(observed.size()), observed_(observed), target_(target[cstr.get()]) {}
    ~AutoMap()
    {
      for (size_t i = observedSize_; i < observed_.size(); ++i)
        target_.push_back(observed_[i].get());
    }

  private:
    size_t observedSize_;
    LeastSquareSolver::AssignmentVector& observed_;
    LeastSquareSolver::AssignmentPtrVector& target_;
  };
}

namespace tvm::solver::abstract
{
  LeastSquareSolver::LeastSquareSolver(bool verbose)
    : objSize_(-1)
    , eqSize_(-1)
    , ineqSize_(-1)
    , buildInProgress_(false)
    , subs_(nullptr)
    , verbose_(verbose)
    , variables_(nullptr)
  {
  }

  void LeastSquareSolver::startBuild(const VariableVector& x, int nObj, int nEq, int nIneq, bool useBounds, const hint::internal::Substitutions* const subs)
  {
    assert(nObj >= 0);
    assert(nEq >= 0);
    assert(nIneq >= 0);

    buildInProgress_ = true;
    variables_ = &x;
    first_.clear();
    for (const auto& xi : variables())
    {
      first_[xi.get()] = {};
    }

    objectiveToAssigments_.clear();
    equalityConstraintToAssigments_.clear();
    inequalityConstraintToAssigments_.clear();
    boundToAssigments_.clear();
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
    const auto& xi = bound->variables()[0];
    RangePtr range = std::make_shared<Range>(xi->getMappingIn(variables()));

    AutoMap autoMap(bound, assignments_, boundToAssigments_);
    auto& first = first_[xi.get()];
    addBound_(bound, range, first.empty());
    first.push_back(bound.get());
  }

  void LeastSquareSolver::addConstraint(LinearConstraintPtr cstr)
  {
    assert(buildInProgress_ && "Attempting to add a constraint without calling startBuild first");

    if (cstr->isEquality())
    {
      AutoMap autoMap(cstr, assignments_, equalityConstraintToAssigments_);
      addEqualityConstraint_(cstr);
      eqSize_ += constraintSize(*cstr);
    }
    else
    {
      AutoMap autoMap(cstr, assignments_, inequalityConstraintToAssigments_);
      addIneqalityConstraint_(cstr);
      ineqSize_ += constraintSize(*cstr);
    }
  }

  void LeastSquareSolver::addObjective(LinearConstraintPtr obj, SolvingRequirementsPtr req, double additionalWeight)
  {
    assert(req->priorityLevel().value() != 0);
    assert(buildInProgress_ && "Attempting to add an objective without calling startBuild first");
    
    if (req->violationEvaluation().value() != requirements::ViolationEvaluationType::L2)
    {
      throw std::runtime_error("[LeastSquareSolver::addObjective]: least-squares only support L2 norm for violation evaluation");
    }
    AutoMap autoMap(obj, assignments_, objectiveToAssigments_);
    addObjective_(obj, req, additionalWeight);
    objSize_ += obj->size();
  }


  void LeastSquareSolver::setMinimumNorm()
  {
    assert(nObj_ == variables().totalSize());
    if (!buildInProgress_)
    {
      throw std::runtime_error("[LeastSquareSolver]: attempting to add an objective without calling startBuild first");
    }
    setMinimumNorm_();
  }

  bool LeastSquareSolver::solve()
  {
    if (buildInProgress_)
    {
      throw std::runtime_error("[LeastSquareSolver]: attempting to solve while in build mode");
    }

    preAssignmentProcess_();
    for (auto& a : assignments_)
      a->assignment.run();
    postAssignmentProcess_();

    if (verbose_)
      printProblemData_();

    bool b = solve_();

    if (verbose_ || !b)
    {
      printDiagnostic_();
      if (verbose_)
      {
        std::cout << "[LeastSquareSolver::solve] solution: " << result_().transpose() << std::endl;
      }
    }

    return b;
  }

  const Eigen::VectorXd& LeastSquareSolver::result() const
  {
    return result_();
  }

  int LeastSquareSolver::constraintSize(const constraint::abstract::LinearConstraint& c) const
  {
    if (c.type() != constraint::Type::DOUBLE_SIDED || handleDoubleSidedConstraint_())
    {
      return c.size();
    }
    else
    {
      return 2 * c.size();
    }
  }

  void LeastSquareSolver::process(const internal::SolverEvents& se)
  {
    bool skipForDebug = false;
    if (skipForDebug) return;
    updateWeights(se);
    bool needMappingUpdate = updateVariables(se);
    auto impactRemove = processRemovedConstraints(se); //ImpactFromChanges impactRemove;
    auto impactAdd = previewAddedConstraints(se);
    auto impactResize = resize_(nObj_, nEq_, nIneq_, !first_.empty());

    if (needMappingUpdate)
    {
      for (auto& a : assignments_)
        a->assignment.onUpdatedMapping(variables(), false);
    }

    // Update the target ranges when needed.
    // We could be much more fine grain as added constraints could be placed last,
    // and not update existing constraints if only constraints are added but this
    // makes things much more complex.
    auto impact = impactRemove || impactAdd;
    auto updateTargetRange = [](const auto& map, int& cumSize, auto rangeProvider, auto sizeProvider)
    {
      cumSize = 0;
      for (const auto& ca : map)
      {
        Range r = rangeProvider(*ca.first);
        for (auto& a : ca.second)
          a->assignment.target().range().start = r.start;
        cumSize += sizeProvider(*ca.first);
      }
    };

    if (impact.equalityConstraints_)
      updateTargetRange(equalityConstraintToAssigments_,
                        eqSize_,
                        [&](const auto& c) { return nextEqualityConstraintRange_(c); },
                        [&](const auto& c) { return constraintSize(c); });

    if (impact.inequalityConstraints_)
      updateTargetRange(inequalityConstraintToAssigments_,
                        ineqSize_,
                        [&](const auto& c) { return nextInequalityConstraintRange_(c); },
                        [&](const auto& c) { return constraintSize(c); });

    int dummy;
    if (impact.bounds_)
      updateTargetRange(boundToAssigments_, 
                        dummy, 
                        [&](const auto& b) { return b.variables()[0]->getMappingIn(variables()); },
                        [](const auto& b) { return 0; });

    if (impact.objectives_)
      updateTargetRange(objectiveToAssigments_,
                        objSize_,
                        [&](const auto& c) { return nextObjectiveRange_(c); },
                        [&](const auto& c) { return c.size(); });

    // Update the matrices and vectors of the target when needed
    auto updateTargetData = [](const auto& map, auto updateFn)
    {
      for (const auto& ca : map)
      {
        for (auto& a : ca.second)
          updateFn(a->assignment.target());
      }
    };

    if (impactResize.equalityConstraints_)
      updateTargetData(equalityConstraintToAssigments_, [&](auto& target) { updateEqualityTargetData(target); });

    if (impactResize.inequalityConstraints_)
      updateTargetData(inequalityConstraintToAssigments_, [&](auto& target) { updateInequalityTargetData(target); });

    if (impactResize.bounds_)
      updateTargetData(boundToAssigments_, [&](auto& target) { updateBoundTargetData(target); });

    if (impactResize.objectives_)
      updateTargetData(objectiveToAssigments_, [&](auto& target) { updateObjectiveTargetData(target); });

    // Final update of the impacted mapping
    if (needMappingUpdate)
    {
      for (auto& a : assignments_)
        a->assignment.onUpdatedTarget(); // onUpdateTarget also takes into account the mapping change due to variable.
    }
    else
    {
      impact.orAssign(impactResize);
      auto updateMapping = [](const auto& map)
      {
        for (const auto& ca : map)
        {
          for (auto& a : ca.second)
            a->assignment.onUpdatedTarget();
        }
      };

      applyImpactLogic(impact);
      if (impact.equalityConstraints_) updateMapping(equalityConstraintToAssigments_);
      if (impact.inequalityConstraints_) updateMapping(inequalityConstraintToAssigments_);
      if (impact.bounds_) updateMapping(boundToAssigments_);
      if (impact.objectives_) updateMapping(objectiveToAssigments_);
    }

    processAddedConstraints(se);
  }

  void LeastSquareSolver::updateWeights(const internal::SolverEvents& se)
  {
    const auto& we = se.weightEvents();

    for (const auto& e : we)
    {
      auto [c, scalar, vector] = e;

      // We might have change of weights on constraints that were not added yet (because process of
      // of weight arise before processing added constraints - if we'd process the weight after, we
      // could have change of weight on removed constraints). Therefore we need to ignore constraints
      // that were not found in the map.
      auto it = objectiveToAssigments_.find(c);
      if (it == objectiveToAssigments_.end()) continue;
      auto& assignments = it->second;

      for (auto& a : assignments)
      {
        bool needReprocess = (scalar && !a->assignment.changeScalarWeightIsAllowed())
                          || (vector && !a->assignment.changeVectorWeightIsAllowed());
        if (needReprocess)
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

  bool LeastSquareSolver::updateVariables(const internal::SolverEvents& se)
  {
    for (const auto& v : se.removedVariables_)
    {
      first_.erase(v.get());
    }
    for (const auto& v : se.addedVariables_)
    {
      first_[v.get()] = {};
    }
    return !(se.removedVariables_.empty() && se.addedVariables_.empty());
  }

  LeastSquareSolver::ImpactFromChanges LeastSquareSolver::processRemovedConstraints(const internal::SolverEvents& se)
  {
    for (const auto& c: se.removedConstraints_)
    {
      if (c->isEquality())
      {
        nEq_ -= constraintSize(*c);
        const auto& assignments = equalityConstraintToAssigments_[c.get()];
        for (auto& a : assignments)
          a->markedForRemoval = true;
        equalityConstraintToAssigments_.erase(c.get());
      }
      else
      {
        nIneq_ -= constraintSize(*c);
        const auto& assignments = inequalityConstraintToAssigments_[c.get()];
        for (auto& a : assignments)
          a->markedForRemoval = true;
        inequalityConstraintToAssigments_.erase(c.get());
      }
    }

    for (const auto& o : se.removedObjectives_)
    {
      nObj_ -= o->size();
      const auto& assignments = objectiveToAssigments_[o.get()];
      for (auto& a : assignments)
        a->markedForRemoval = true;
      objectiveToAssigments_.erase(o.get());
    }

    for (const auto& b : se.removedBounds_)
    {
      Variable* xb = b->variables()[0].get();
      auto& first = first_[xb];
      if (first[0] == b.get() && variables().contains(*xb))
      {
        // We need to remove the bound that appears first, and the associated variable
        // is still present in the variable vector. 
        if (first.size() == 1)
        {
          // There will be no more bounds on xb.
          removeBounds_(*xb);
        }
        else
        {
          // We make the next bound constraint as first.
          auto nextBound = first[1];
          auto& nextBoundAssignments = boundToAssigments_[nextBound];
          for (auto& a : nextBoundAssignments)
            a->assignment = tvm::scheme::internal::Assignment::reprocess(a->assignment, b->variables()[0], true);
        }
        first.erase(first.begin());
      }
      else
      {
        auto it = std::find(first.begin() + 1, first.end(), b.get());
        first.erase(it);
      }

      const auto& assignments = boundToAssigments_[b.get()];
      for (auto& a : assignments)
        a->markedForRemoval = true;
      boundToAssigments_.erase(b.get());
    }

    //clear assignements
    auto it = std::remove_if(assignments_.begin(), assignments_.end(), [](const auto& it) {return it->markedForRemoval; });
    assignments_.erase(it, assignments_.end());

    return { !se.removedConstraints_.empty(), !se.removedConstraints_.empty(), !se.removedBounds_.empty(), !se.removedObjectives_.empty() };
  }

  LeastSquareSolver::ImpactFromChanges LeastSquareSolver::previewAddedConstraints(const internal::SolverEvents& se)
  {
    // When used as part of process, the following code will generate target ranges
    // outside of the matrices bounds. It relies on the fact that the mapping of the
    // assignment is updated after.
    eqSize_ = nEq_;
    ineqSize_ = nIneq_;
    objSize_ = nObj_;
    for (const auto& c : se.addedConstraints_)
    {
      if (c->isEquality())
        nEq_ += constraintSize(*c);
      else
        nIneq_ += constraintSize(*c);
    }

    for (const auto& o : se.addedObjectives_)
      nObj_ += o.c->size();

    bool impactEq = nEq_ != eqSize_;
    bool impactIneq = nIneq_ != ineqSize_;
    return { impactEq, impactIneq, false, !se.addedObjectives_.empty() };
  }

  void LeastSquareSolver::processAddedConstraints(const internal::SolverEvents& se)
  {
    buildInProgress_ = true;
    for (const auto& c : se.addedConstraints_)
      addConstraint(c);

    for (const auto& b : se.addedBounds_)
      addBound(b);

    for (const auto& o : se.addedObjectives_)
      addObjective(o.c, o.req, o.scalarizationWeight);

    assert(nObj_ == objSize_);
    assert(nEq_ == eqSize_);
    assert(nIneq_ == ineqSize_);
    buildInProgress_ = false;
  }

  LeastSquareSolver::ImpactFromChanges LeastSquareSolver::ImpactFromChanges::operator||(bool b)
  {
    return { this->equalityConstraints_ || b,
             this->inequalityConstraints_ || b,
             this->bounds_ || b,
             this->objectives_ || b };
  }

  LeastSquareSolver::ImpactFromChanges LeastSquareSolver::ImpactFromChanges::operator||(const ImpactFromChanges& other)
  {
    return { this->equalityConstraints_ || other.equalityConstraints_,
             this->inequalityConstraints_ || other.inequalityConstraints_,
             this->bounds_ || other.bounds_,
             this->objectives_ || other.objectives_ };
  }

  LeastSquareSolver::ImpactFromChanges& LeastSquareSolver::ImpactFromChanges::orAssign(const ImpactFromChanges& other)
  {
    *this = *this || other;
    return *this;
  }

  void LeastSquareSolver::applyImpactLogic(ImpactFromChanges& impact)
  {
    // do nothing;
  }
}