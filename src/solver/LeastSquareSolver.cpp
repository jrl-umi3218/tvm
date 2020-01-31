#include <tvm/solver/abstract/LeastSquareSolver.h>

#include <tvm/VariableVector.h>

namespace tvm
{

namespace solver
{

namespace abstract
{
  LeastSquareSolver::LeastSquareSolver()
    : objSize_(-1)
    , eqSize_(-1)
    , ineqSize_(-1)
    , buildInProgress_(false)
    , subs_(nullptr)
  {
  }

  void LeastSquareSolver::startBuild(const VariableVector& x, int m1, int me, int mi, bool useBounds, const hint::internal::Substitutions& subs)
  {
    assert(m1 >= 0);
    assert(me >= 0);
    assert(mi >= 0);

    buildInProgress_ = true;
    variables_ = &x;
    first_.clear();
    for (const auto& xi : variables())
    {
      first_[xi.get()] = true;
    }

    subs_ = &subs;

    initializeBuild_(m1, me, mi, useBounds);
    me_ = me;
    mi_ = mi;
    m1_ = m1;
    objSize_ = 0;
    eqSize_ = 0;
    ineqSize_ = 0;
  }

  void LeastSquareSolver::finalizeBuild()
  {
    buildInProgress_ = false;
  }

  void LeastSquareSolver::addBound(LinearConstraintPtr bound)
  {
    if (!buildInProgress_)
    {
      throw std::runtime_error("[LeastSquareSolver]: attempting to add a bound without calling startBuild first");
    }
    assert(bound->variables().numberOfVariables() == 1 && "A bound constraint can be only on one variable.");
    const auto& xi = bound->variables()[0];
    RangePtr range = std::make_shared<Range>(xi->getMappingIn(variables())); //FIXME: for now we do not keep a pointer on the range nor the target.

    AutoMap autoMap(bound, assignments_, boundToAssigments_);
    bool& first = first_[xi.get()];
    addBound_(bound, range, first);
    first = false;
  }

  void LeastSquareSolver::addConstraint(LinearConstraintPtr cstr)
  {
    if (!buildInProgress_)
    {
      throw std::runtime_error("[LeastSquareSolver]: attempting to add a constraint without calling startBuild first");
    }
    AutoMap autoMap(cstr, assignments_, constraintToAssigments_);
    if (cstr->isEquality())
    {
      addEqualityConstraint_(cstr);
      eqSize_ += constraintSize(cstr);
    }
    else
    {
      addIneqalityConstraint_(cstr);
      ineqSize_ += constraintSize(cstr);
    }
  }

  void LeastSquareSolver::addObjective(LinearConstraintPtr obj, const SolvingRequirementsPtr req, double additionalWeight)
  {
    assert(req->priorityLevel().value() != 1);
    if (!buildInProgress_)
    {
      throw std::runtime_error("[LeastSquareSolver]: attempting to add an objective without calling startBuild first");
    }
    if (req->violationEvaluation().value() != requirements::ViolationEvaluationType::L2)
    {
      throw std::runtime_error("[LeastSquareSolver::addObjective]: least-squares only support L2 norm for violation evaluation");
    }
    addObjective_(obj, req, additionalWeight);
  }


  bool LeastSquareSolver::solve()
  {
    if (buildInProgress_)
    {
      throw std::runtime_error("[LeastSquareSolver]: attempting to solve while in build mode");
    }

    preAssignmentProcces_();
    for (auto& a : assignments_)
      a.run();
    postAssignementProcess_();

    return solve_();
  }

  int LeastSquareSolver::constraintSize(const LinearConstraintPtr& c) const
  {
    if (handleDoubleSidedConstraint_())
    {
      return c->size();
    }
    else
    {
      return 2 * c->size();
    }
  }
}

}

}