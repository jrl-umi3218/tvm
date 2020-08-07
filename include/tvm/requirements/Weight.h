/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/


#pragma once

#include <tvm/requirements/abstract/SingleSolvingRequirement.h>

namespace tvm
{

namespace requirements
{

  /** This class represents the scalar weight alpha of a constraint,
    * within its priority level. It is meant to ajust the influence of
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

    WeightBase(double alpha)
      : abstract::SingleSolvingRequirement<double, Lightweight>(alpha, false)
    {
      if (alpha < 0)
        throw std::runtime_error("weight must be non negative.");
    }

    DEFINE_LW_NON_LW_CONVERSION_OPERATORS(WeightBase, double, Lightweight)
  };

  using Weight = WeightBase<true>;

}  // namespace requirements

}  // namespace tvm
