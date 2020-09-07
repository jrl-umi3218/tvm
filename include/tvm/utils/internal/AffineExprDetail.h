/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>

#include <Eigen/Core>

namespace tvm
{
namespace utils
{
namespace internal
{

/** Type of Eigen::MatrixXd::Identity() */
using IdentityType = decltype(Eigen::MatrixXd::Identity());
/** Type of double*Eigen::MatrixXd::Identity() */
using MultIdentityType = decltype(2. * Eigen::MatrixXd::Identity());
/** Type of -Eigen::MatrixXd::Identity() */
using MinusIdentityType = decltype(-Eigen::MatrixXd::Identity());

/** A dummy class to represent the absence of constant part in an affine expression. */
class NoConstant
{
public:
  NoConstant operator-() const { return {}; }
};

/** Adding an absent constant part with an existing constant part. */
template<typename RhsType>
inline const RhsType & operator+(const NoConstant &, const Eigen::MatrixBase<RhsType> & rhs)
{
  return rhs.derived();
}

/** Adding an existing constant part with an absent constant part. */
template<typename LhsType>
inline const LhsType & operator+(const Eigen::MatrixBase<LhsType> & lhs, const NoConstant &)
{
  return lhs.derived();
}

/** Adding two absent constant parts. */
inline auto operator+(const NoConstant &, const NoConstant &) { return NoConstant(); }

/** Overload for post-multiplying by NoConstant. In this case, we need to return NoConstant.*/
template<typename MultType>
inline NoConstant operator*(const MultType & /*m*/, const NoConstant &)
{
  return {};
}

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
struct RefSelector
{
  using type = typename Eigen::internal::ref_selector<Derived>::type;
};
template<>
struct RefSelector<NoConstant>
{
  using type = NoConstant;
};

template<typename Derived>
using RefSelector_t = typename RefSelector<Derived>::type;

/** Result type for the addition of two constant parts, existing or not.*/
template<typename LhsType, typename RhsType>
using AddConstantsRetType =
    std::remove_const_t<std::remove_reference_t<decltype(std::declval<LhsType>() + std::declval<RhsType>())>>;

/** Taking the opposite of each element i of the input tuple where the i's
 * are the elements given by the sequence of index.
 */
template<typename Tuple, size_t... Indices>
auto tupleUnaryMinus(const Tuple & tuple, std::index_sequence<Indices...>)
{
  return std::make_tuple((-std::get<Indices>(tuple))...);
}

/** Premultiplication by m of each element i of the input tuple where the i's
 * are the elements given by the sequence of index.
 */
template<typename MultType, typename Tuple, size_t... Indices>
auto tuplePremult(const MultType & m, const Tuple & tuple, std::index_sequence<Indices...>)
{
  return std::make_tuple((m * std::get<Indices>(tuple))...);
}
} // namespace internal
} // namespace utils
} // namespace tvm

namespace Eigen
{
namespace internal
{
/** This is required to get GCC to compile the SFINAE constructors. */
template<>
struct traits<tvm::utils::internal::NoConstant> : public traits<tvm::utils::internal::IdentityType>
{
};
} // namespace internal
} // namespace Eigen
