/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/requirements/abstract/SingleSolvingRequirement.h>

namespace tvm
{

namespace requirements
{

/** This class represents the scalar weight alpha of a constraint,
 * within its priority level. It is meant to adjust the influence of
 * several constraints at the same level.
 *
 * Given a scalar weight \p alpha, and a constraint violation measurement
 * f(x), the product alpha*f(x) will be minimized.
 *
 * By default the weight is 1.
 */
template<bool Lightweight = true>
class WeightBase : public abstract::SingleSolvingRequirement<double, Lightweight>
{
public:
  /** Default weight = 1*/
  WeightBase() : abstract::SingleSolvingRequirement<double, Lightweight>(1.0, true) {}

  WeightBase(double alpha) : abstract::SingleSolvingRequirement<double, Lightweight>(alpha, false)
  {
    if(alpha < 0)
      throw std::runtime_error("weight must be non negative.");
  }

  TVM_DEFINE_LW_NON_LW_CONVERSION_OPERATORS(WeightBase, double, Lightweight)
};

using Weight = WeightBase<true>;

} // namespace requirements

} // namespace tvm
