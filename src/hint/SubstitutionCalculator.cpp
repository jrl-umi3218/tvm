/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/hint/abstract/SubstitutionCalculator.h>

namespace tvm
{

namespace hint
{

namespace abstract
{

std::unique_ptr<SubstitutionCalculatorImpl> SubstitutionCalculator::impl(const std::vector<LinearConstraintPtr> & cstr,
                                                                         const std::vector<VariablePtr> & x,
                                                                         int rank) const
{
  return impl_(cstr, x, rank);
}

} // namespace abstract

} // namespace hint

} // namespace tvm
