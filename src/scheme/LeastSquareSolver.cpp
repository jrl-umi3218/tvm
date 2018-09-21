#include <tvm/scheme/abstract/LeastSquareSolver.h>

#include <tvm/VariableVector.h>

namespace tvm
{

namespace scheme
{

namespace abstract
{
  void LeastSquareSolver::reserve(const VariableVector& x, int m0, int me, int mi, bool useBounds)
  {
    x_ = &x;
    first_.clear();
    for (const auto& xi : x.variables())
    {
      first_[xi.get()] = true;
    }

    reserve_(m0, me, mi, useBounds);
  }

  void LeastSquareSolver::addBound(LinearConstraintPtr bound)
  {
    const auto& xi = bound->variables()[0];
    RangePtr range = std::make_shared<Range>(xi->getMappingIn(*x_)); //FIXME: for now we do not keep a pointer on the range nor the target.
    AssignmentTarget target(range, memory->l, memory->u);
    memory->assignments.emplace_back(Assignment(b.constraint, target, xi, first[xi.get()]));
    first[xi.get()] = false;
  }
}

}

}