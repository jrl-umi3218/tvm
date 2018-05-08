#include <tvm/hint/internal/AutoCalculator.h>
#include <tvm/constraint/abstract/LinearConstraint.h>
#include <tvm/hint/internal/DiagonalCalculator.h>
#include <tvm/hint/internal/GenericCalculator.h>

namespace tvm
{

namespace hint
{

namespace internal
{

  std::unique_ptr<abstract::SubstitutionCalculatorImpl> AutoCalculator::impl_(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank) const
  {
    if (cstr.size() > 1 || x.size() > 1)
    {
      return GenericCalculator().impl(cstr, x, rank);
    }
    else
    {
      const auto& p = cstr[0]->jacobian(*x[0]).properties();
      if (p.isDiagonal() && p.isInvertible())
      {
        return DiagonalCalculator().impl(cstr, x, rank);
      }
      return GenericCalculator().impl(cstr, x, rank);
    }
  }

}

}

}
