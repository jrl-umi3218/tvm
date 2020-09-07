/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/hint/abstract/SubstitutionCalculatorImpl.h>

#include <memory>
#include <vector>

namespace tvm
{

namespace hint
{

namespace abstract
{
/** A SubstitutionCalculator is a lightweight factory that can generates a
 * SubstitutionCalculatorImpl.
 * It is used to specify custom operations to be made during the substitution
 * process.
 */
class TVM_DLLAPI SubstitutionCalculator
{
public:
  virtual ~SubstitutionCalculator() = default;

  std::unique_ptr<SubstitutionCalculatorImpl> impl(const std::vector<LinearConstraintPtr> & cstr,
                                                   const std::vector<VariablePtr> & x,
                                                   int rank) const;

protected:
  virtual std::unique_ptr<SubstitutionCalculatorImpl> impl_(const std::vector<LinearConstraintPtr> & cstr,
                                                            const std::vector<VariablePtr> & x,
                                                            int rank) const = 0;
};

} // namespace abstract

} // namespace hint

} // namespace tvm
