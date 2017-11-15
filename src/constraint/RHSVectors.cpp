#include <tvm/constraint/internal/RHSVectors.h>

namespace tvm
{
namespace constraint
{
namespace internal
{
  RHSVectors::RHSVectors(Type ct, RHS cr)
    : usel_((ct == Type::GREATER_THAN || ct == Type::DOUBLE_SIDED) && cr != RHS::ZERO)
    , useu_((ct == Type::LOWER_THAN || ct == Type::DOUBLE_SIDED) && cr != RHS::ZERO)
    , usee_(ct == Type::EQUAL && cr != RHS::ZERO)
  {

  }

  void RHSVectors::resize(int n)
  {
    if (usel_)
      l_.resize(n);

    if (useu_)
      u_.resize(n);

    if (usee_)
      e_.resize(n);
  }
}
}
}