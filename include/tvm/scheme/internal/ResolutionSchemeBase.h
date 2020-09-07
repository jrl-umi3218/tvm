/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/internal/ObjWithId.h>
#include <tvm/scheme/internal/SchemeAbilities.h>

namespace tvm
{

namespace scheme
{
using identifier = int;

namespace internal
{
/** Base class for solving a ControlProblem*/
class TVM_DLLAPI ResolutionSchemeBase : public tvm::internal::ObjWithId
{
public:
  double big_number() const;
  void big_number(double big);

protected:
  /** Constructor, meant only for derived classes
   *
   * \param abilities The set of abilities for this scheme.
   * \param big A big number use to represent infinity, in particular when
   * specifying non-existing bounds (e.g. x <= Inf is given as x <= big).
   */
  ResolutionSchemeBase(SchemeAbilities abilities, double big = constant::big_number);

  SchemeAbilities abilities_;

  /** A number to use for infinite bounds*/
  double big_number_;
};
} // namespace internal

} // namespace scheme

} // namespace tvm
