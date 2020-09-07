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
#include <tvm/constraint/enums.h>
#include <tvm/defs.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/function/abstract/Function.h>

#include <Eigen/Core>

namespace tvm
{

namespace utils
{

namespace internal
{
/** Describe the type of a right-hand-side argument.
 * - Zero: the rhs is the zero vector
 * - Double: the rhs is a vector with all elements equal to a given double
 * - Vector: the rhs is a general vector
 */
enum class RHSType
{
  Zero,
  Double,
  Vector
};

/** A union-like class, that can represent nothing, a double or a vector.
 * Which of these 3 options is described by its RHSType
 */
class RHS
{
public:
  RHS(double d);
  template<typename Derived>
  RHS(const Eigen::MatrixBase<Derived> & v);

  Eigen::VectorXd toVector(Eigen::DenseIndex n) const;

  RHSType type_;
  double d_;
  Eigen::VectorXd v_;
};

inline RHS::RHS(double d) : type_(d == 0 ? RHSType::Zero : RHSType::Double), d_(d) {}

template<typename Derived>
inline RHS::RHS(const Eigen::MatrixBase<Derived> & v) : type_(RHSType::Vector), v_(v)
{
}

inline Eigen::VectorXd RHS::toVector(Eigen::DenseIndex n) const
{
  switch(type_)
  {
    case tvm::utils::internal::RHSType::Zero:
      return Eigen::VectorXd::Zero(n);
    case tvm::utils::internal::RHSType::Double:
      return Eigen::VectorXd::Constant(n, d_);
    case tvm::utils::internal::RHSType::Vector:
      assert(v_.size() == n);
      return v_;
    default:
      assert(false);
      return Eigen::VectorXd();
      break;
  }
}

} // namespace internal

} // namespace utils

} // namespace tvm
