/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>

#include <tvm/hint/abstract/SubstitutionCalculator.h>

namespace tvm
{

namespace hint
{

namespace internal
{

/** Automatically generates the most appropriate calculator for the given
 * constraints and variables.
 *
 * Current rules:
 * - for a simple substitution with invertible diagonal matrix, generates a
 *   DiagonalCalculator
 * - otherwise generates a GenericCalculator
 *
 * \note You need to ensure that the matrix properties used when applying the
 * rules have been correctly set. If this is not the case, this should be
 * corrected at the function level. In particular it is improper to rely on
 * a run of the update pipeline to have all the properties correctly set.
 */
class TVM_DLLAPI AutoCalculator : public abstract::SubstitutionCalculator
{
protected:
  std::unique_ptr<abstract::SubstitutionCalculatorImpl> impl_(const std::vector<LinearConstraintPtr> & cstr,
                                                              const std::vector<VariablePtr> & x,
                                                              int rank) const override;
};

} // namespace internal

} // namespace hint

} // namespace tvm
