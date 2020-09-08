/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

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
  RHS(const Eigen::MatrixBase<Derived> & v);

  Eigen::VectorXd toVector(Eigen::DenseIndex n) const;

  RHSType type_;
  double d_;
  Eigen::VectorXd v_;
};

inline RHS::RHS(double d) : type_(d == 0 ? RHSType::Zero : RHSType::Double), d_(d) {}

template<typename Derived>
inline RHS::RHS(const Eigen::MatrixBase<Derived> & v) : type_(RHSType::Vector), v_(v)
{}

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
