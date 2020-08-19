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
    {}

    ViolationEvaluationBase(ViolationEvaluationType t) 
      : abstract::SingleSolvingRequirement<ViolationEvaluationType, Lightweight>(t, false) 
    {}

    TVM_DEFINE_LW_NON_LW_CONVERSION_OPERATORS(ViolationEvaluationBase, ViolationEvaluationType, Lightweight)
  };

  using ViolationEvaluation = ViolationEvaluationBase<true>;

}  // namespace requirements

}  // namespace tvm
