#include <tvm/scheme/internal/ResolutionSchemeBase.h>

#include <tvm/scheme/internal/IdProvider.h>

namespace tvm
{

namespace scheme
{

namespace internal
{

  IdProvider ResolutionSchemeBase::idProvider_;

  ResolutionSchemeBase::ResolutionSchemeBase(SchemeAbilities abilities, double big)
    : abilities_(abilities)
    , big_number_(big)
    , id_(ResolutionSchemeBase::idProvider_.makeId())
  {
    assert(big > 0);
  }

  identifier ResolutionSchemeBase::id() const
  {
    return id_;
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
