#include <tvm/scheme/LSSOLLeastSquareSolver.h>

#include <tvm/scheme/internal/AssignmentTarget.h>

namespace tvm
{

namespace scheme
{
  void LSSOLLeastSquareSolver::reserve_(int m0, int me, int mi, bool useBounds)
  {

  }

  void LSSOLLeastSquareSolver::addBound_(LinearConstraintPtr bound, RangePtr range, bool first)
  {
    internal::AssignmentTarget target(range, l_, u_);
  }
}

}