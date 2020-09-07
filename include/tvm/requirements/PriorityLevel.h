/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/requirements/abstract/SingleSolvingRequirement.h>

namespace tvm
{

namespace requirements
{

/** This class represents the priority level of a constraint.
 * The default priority level is 0
 */
template<bool Lightweight = true>
class PriorityLevelBase : public abstract::SingleSolvingRequirement<int, Lightweight>
{
public:
  /** Default constructor p=0 */
  PriorityLevelBase() : abstract::SingleSolvingRequirement<int, Lightweight>(0, true) {}

  /** Priority level p>=0*/
  PriorityLevelBase(int p) : abstract::SingleSolvingRequirement<int, Lightweight>(p, false)
  {
    if(p < 0)
      throw std::runtime_error("Priority level must be non-negative.");
  }

  TVM_DEFINE_LW_NON_LW_CONVERSION_OPERATORS(PriorityLevelBase, int, Lightweight)
};

using PriorityLevel = PriorityLevelBase<true>;

} // namespace requirements

} // namespace tvm
