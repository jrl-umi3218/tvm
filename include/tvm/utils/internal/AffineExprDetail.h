/* Copyright 2017-2019 CNRS-AIST JRL and CNRS-UM LIRMM
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

#include <eigen/Core>

namespace tvm
{
namespace utils
{
namespace internal
{
  /** Shortcut to an internal Eigen type to store expressions or matrices.
    * 
    * When keeping internally a reference to an Eigen object, we need to have different behaviors
    * depending on wether the object has a large memory and should not be copied or is a 
    * lightweight proxy.
    * For matrix, we need to keep a const ref, while for matrix expression  we need to take a copy
    * of the expression.This is exactly the purpose of ref_selector, that is used to this effect
    * in e.g.CWiseBinaryOp.
    */
  template<typename Derived>
  using RefSelector_t = typename Eigen::internal::ref_selector<Derived>::type;

  /** A dummy class to represent the absence of constant part in an affine expression. */
  class NoConstant {};

  /** Adding two existing constant parts together.*/
  template<typename LhsType, typename RhsType>
  inline auto addConstants(const Eigen::MatrixBase<LhsType>& lhs, const Eigen::MatrixBase<RhsType>& rhs) { return lhs.derived() + rhs.derived(); }

  /** Adding an absent constant part with an existing constant part. */
  template<typename RhsType>
  inline const RhsType& addConstants(const NoConstant&, const Eigen::MatrixBase<RhsType>& rhs) { return rhs.derived(); }

  /** Adding an existing constant part with an absent constant part. */
  template<typename LhsType>
  inline const LhsType& addConstants(const Eigen::MatrixBase<LhsType>& lhs, const NoConstant&) { return lhs.derived(); }

  /** Adding two absent constant parts. */
  inline auto addConstants(const NoConstant&, const NoConstant&) { return NoConstant(); }

  /** Result type for the addition of two constant parts, existing or not.*/
  template<typename LhsType, typename RhsType>
  using AddConstantsRetType = std::remove_const_t<std::remove_reference_t<decltype(addConstants(std::declval<LhsType>(), std::declval<RhsType>()))> >;
}
}
}