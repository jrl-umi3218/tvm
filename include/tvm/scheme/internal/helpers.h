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

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/constraint/enums.h>

#include <vector>

namespace tvm
{
namespace hint
{
namespace internal
{
  class Substitutions;
}
}
namespace function
{
  class BasicLinearFunction;
}

namespace scheme
{
namespace internal
{
  class ProblemComputationData;

  /** Get the computation data linked to a particular resolution scheme.
    * If this data does not exist, create it, using the resolution scheme as
    * a factory
    * \param The resolution scheme for which we want to get back the
    * computation data. It needs to have a method \a createComputationData.
    */
  template<typename Problem, typename Scheme>
  inline ProblemComputationData* getComputationData(Problem& problem, const Scheme& resolutionScheme);

  /** We consider as bound a constraint with a single variable and a diagonal,
    * invertible jacobian.
    * It would be possible to accept non-invertible sparse diagonal jacobians
    * as well, in which case the zero elements of the diagonal would 
    * correspond to non-existing bounds, but this requires quite a lot of
    * work for something that is unlikely to happen and could be expressed
    * by changing the bound itself to +/- infinity.
    */
  bool TVM_DLLAPI isBound(const ConstraintPtr& c);

  /** Assert if a constraint is a bound in the presence of substitutions
    * \param c the constraint
    * \param subs the set of substitutions
    * We have a bound if c is a bound and the variable is not substituted or it
    * is by an expression with a single varaible and an invertible, diagonal
    * jacobian.
    */
  bool TVM_DLLAPI isBound(const ConstraintPtr& c, const hint::internal::Substitutions& subs);

  /** Assert if a constraint is a bound in the presence of substitutions
    * \param c the constraint
    * \param x the set of variables being substituted
    * \param xsub the set of substitution functions corresponding to those variables.
    * We have a bound if c is a bound and the variable is not substituted or it
    * is by an expression with a single varaible and an invertible, diagonal
    * jacobian.
    */
  bool TVM_DLLAPI isBound(const ConstraintPtr& c, const std::vector<VariablePtr>& x,
                          const std::vector<std::shared_ptr<function::BasicLinearFunction>>& xsub);

  /** Check if the constraint can be used as a bound for a given target convention.
    * For example l <= x <= u is a bound, but it cannot be transformed into x >= lb.
    */
  bool TVM_DLLAPI canBeUsedAsBound(const ConstraintPtr& c, const hint::internal::Substitutions& subs,
                                   constraint::Type targetConvention);

  /** Version with separated susbtituted variables and substitution functions.*/
  bool TVM_DLLAPI canBeUsedAsBound(const ConstraintPtr& c, const std::vector<VariablePtr>& x,
                                   const std::vector<std::shared_ptr<function::BasicLinearFunction>>& xsub,
                                   constraint::Type targetConvention);
}
}
}