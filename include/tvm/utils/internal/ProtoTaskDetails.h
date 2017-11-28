#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/constraint/enums.h>
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
    RHS(const Eigen::MatrixBase<Derived>& v);

    Eigen::VectorXd toVector(Eigen::DenseIndex n) const;

    RHSType type_;
    double d_;
    Eigen::VectorXd v_;
  };


  inline RHS::RHS(double d)
    : type_(d == 0 ? RHSType::Zero : RHSType::Double)
    , d_(d)
  {
  }

  template<typename Derived>
  inline RHS::RHS(const Eigen::MatrixBase<Derived>& v)
    : type_(RHSType::Vector)
    , v_(v)
  {
  }


  inline Eigen::VectorXd RHS::toVector(Eigen::DenseIndex n) const
  {
    switch (type_)
    {
    case tvm::utils::internal::RHSType::Zero: return Eigen::VectorXd::Zero(n);
    case tvm::utils::internal::RHSType::Double: return Eigen::VectorXd::Constant(n, d_);
    case tvm::utils::internal::RHSType::Vector: assert(v_.size() == n); return v_;
    default: assert(false); return Eigen::VectorXd(); break;
    }
  }

} // namespace internal

} // namespace utils

} // namespace tvm
