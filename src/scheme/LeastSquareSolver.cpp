#include <tvm/scheme/abstract/LeastSquareSolver.h>

#include <tvm/VariableVector.h>

namespace tvm
{

namespace scheme
{

namespace abstract
{
  void LeastSquareSolver::startBuild(int m0, int me, int mi, bool useBounds, const hint::internal::Substitutions& subs)
  {
    buildInProgress_ = true;
    first_.clear();
    for (const auto& xi : variables())
    {
      first_[xi.get()] = true;
    }

    initializeBuild_(m0, me, mi, useBounds);
  }

  void LeastSquareSolver::finalizeBuild()
  {
    buildInProgress_ = false;
  }

  void LeastSquareSolver::addBound(LinearConstraintPtr bound)
  {
    if (!buildInProgress_)
    {
      throw std::runtime_error("[LeastSquareSolver]: attempting to add bound without calling startBuild first");
    }
    const auto& xi = bound->variables()[0];
    RangePtr range = std::make_shared<Range>(xi->getMappingIn(variables())); //FIXME: for now we do not keep a pointer on the range nor the target.

    addBound_(bound, )
  }
}

}

}