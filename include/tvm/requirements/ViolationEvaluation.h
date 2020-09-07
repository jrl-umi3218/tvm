/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/requirements/abstract/SingleSolvingRequirement.h>

#include <Eigen/Core>

namespace tvm
{

namespace requirements
{

/** Given a constraint, let the vector v(x) be its componentwise
 * violation.
 *
 * For example, for the constraint c(x) = 0, we simply have v(x) = c(x),
 * for c(x) >= b, we have v(x) = max(b-c(x),0).
 *
 * This enumeration specifies how v(x) is made into a scalar measure
 * f(x) of this violation.
 */
enum class ViolationEvaluationType
{
  /** f(x) = sum(abs(v_i(x))) */
  L1,
  /** f(x) = v(x)^T*v(x) */
  L2,
  /** f(x) = max(abs(v_i(x))) */
  LINF
};

/** A class specifying how a constraint violation should be handled.
 * By default the L2 norm of the violation is used.
 * \sa ViolationEvaluationType
 */
template<bool Lightweight = true>
class ViolationEvaluationBase : public abstract::SingleSolvingRequirement<ViolationEvaluationType, Lightweight>
{
public:
  /** Default value: ViolationEvaluationType::L2*/
  ViolationEvaluationBase()
  : abstract::SingleSolvingRequirement<ViolationEvaluationType, Lightweight>(ViolationEvaluationType::L2, true)
  {
  }

  ViolationEvaluationBase(ViolationEvaluationType t)
  : abstract::SingleSolvingRequirement<ViolationEvaluationType, Lightweight>(t, false)
  {
  }

  TVM_DEFINE_LW_NON_LW_CONVERSION_OPERATORS(ViolationEvaluationBase, ViolationEvaluationType, Lightweight)
};

using ViolationEvaluation = ViolationEvaluationBase<true>;

} // namespace requirements

} // namespace tvm
