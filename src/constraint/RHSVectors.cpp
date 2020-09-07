/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/constraint/internal/RHSVectors.h>

namespace tvm
{
namespace constraint
{
namespace internal
{
RHSVectors::RHSVectors(Type ct, RHS cr)
: use_l_((ct == Type::GREATER_THAN || ct == Type::DOUBLE_SIDED) && cr != RHS::ZERO),
  use_u_((ct == Type::LOWER_THAN || ct == Type::DOUBLE_SIDED) && cr != RHS::ZERO),
  use_e_(ct == Type::EQUAL && cr != RHS::ZERO)
{
}

void RHSVectors::resize(int n)
{
  if(use_l_)
    l_.resize(n);

  if(use_u_)
    u_.resize(n);

  if(use_e_)
    e_.resize(n);
}
} // namespace internal
} // namespace constraint
} // namespace tvm
