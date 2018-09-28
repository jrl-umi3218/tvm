#include <tvm/scheme/internal/ResolutionSchemeBase.h>

namespace tvm
{

namespace scheme
{

namespace internal
{

  ResolutionSchemeBase::ResolutionSchemeBase(SchemeAbilities abilities, double big)
    : abilities_(abilities)
    , big_number_(big)
  {
    assert(big > 0);
  }

  double ResolutionSchemeBase::big_number() const
  {
    return big_number_;
  }

  void ResolutionSchemeBase::big_number(double big)
  {
    assert(big > 0);
    big_number_ = big;
  }

}  // namespace abstract

}  // namespace scheme

}  // namespace tvm
