#pragma once

/** This file contains several helpers for properly check and display Eigen objects. */

#include <Eigen/Core>

#include "doctest.h"

namespace Eigen
{
/** \brief Helper class to compare eigen matrices in a fuzzy way in doctest
 *
 * This allows to write e.g. \c CHECK_EQ(A, Eigen::Approx(B)) for 2 eigen matrices to perform the
 * equivalent of \c CHECK_UNARY(B.isApprox(A, epsilon)) but still have the matrices properly
 * displayed in case of error.
 */
class Approx
{
public:
  Approx(const Ref<const MatrixXd> & M) : epsilon_(NumTraits<double>::dummy_precision()), value_(M) {}

  /** Change the epsilon used by isApprox.*/
  Approx & epsilon(double newEpsilon)
  {
    epsilon_ = newEpsilon;
    return *this;
  }

#define EIGEN_APPROX_PREFIX \
  template<typename T>      \
  friend typename std::enable_if<std::is_constructible<Ref<const MatrixXd>, T>::value, bool>::type

  EIGEN_APPROX_PREFIX operator==(const T & lhs, const Approx & rhs) { return rhs.eq(lhs); }
  EIGEN_APPROX_PREFIX operator==(const Approx & lhs, const T & rhs) { return operator==(rhs, lhs); }
  EIGEN_APPROX_PREFIX operator!=(const T & lhs, const Approx & rhs) { return !operator==(lhs, rhs); }
  EIGEN_APPROX_PREFIX operator!=(const Approx & lhs, const T & rhs) { return !operator==(rhs, lhs); }

  friend doctest::String toString(const Approx & in) { return "Approx(" + doctest::toString(in.value_) + ")\n"; }

private:
  template<typename Derived>
  bool eq(const DenseBase<Derived> & other) const
  {
    return value_.isApprox(other, epsilon_);
  }

  double epsilon_;
  const Ref<const MatrixXd> & value_;
};
} // namespace Eigen

/** Specialize the doctest::detail::filldata for an Eigen type T.
 *
 * Compared to classical types, line breaks are inserted before and after the instance
 * representation. Vectors are transposed for display.
 */
#define EIGEN_DOCTEST_FILLDATA(T)                         \
  template<>                                              \
  struct filldata<T>                                      \
  {                                                       \
    static void fill(std::ostream * stream, const T & in) \
    {                                                     \
      if(in.cols() == 1)                                  \
        *stream << "\n" << in.transpose() << "\n";        \
      else                                                \
        *stream << "\n" << in << "\n";                    \
    }                                                     \
  };

/** Generate filldata<Expr<T>> for a list of Expr usually used with Eigen. */
#define EIGEN_DOCTEST_GENERATE_FILLDATA(T)            \
  EIGEN_DOCTEST_FILLDATA(T)                           \
  EIGEN_DOCTEST_FILLDATA(Eigen::Ref<T>)               \
  EIGEN_DOCTEST_FILLDATA(Eigen::Ref<const T>)         \
  EIGEN_DOCTEST_FILLDATA(Eigen::Block<T>)             \
  EIGEN_DOCTEST_FILLDATA(Eigen::Block<const T>)       \
  EIGEN_DOCTEST_FILLDATA(Eigen::Block<Eigen::Ref<T>>) \
  EIGEN_DOCTEST_FILLDATA(Eigen::Block<const Eigen::Ref<const T>>)

/** Generate a list of filldata for a given scalar type \p s*/
#define EIGEN_DOCTEST_GENERATE_FILLDATA_FOR_SCALAR_TYPE(s) \
  EIGEN_DOCTEST_GENERATE_FILLDATA(Eigen::MatrixX##s)       \
  EIGEN_DOCTEST_GENERATE_FILLDATA(Eigen::Matrix2##s)       \
  EIGEN_DOCTEST_GENERATE_FILLDATA(Eigen::Matrix3##s)       \
  EIGEN_DOCTEST_GENERATE_FILLDATA(Eigen::Matrix4##s)

namespace doctest
{
namespace detail
{
EIGEN_DOCTEST_GENERATE_FILLDATA_FOR_SCALAR_TYPE(d)
EIGEN_DOCTEST_GENERATE_FILLDATA_FOR_SCALAR_TYPE(f)
EIGEN_DOCTEST_GENERATE_FILLDATA_FOR_SCALAR_TYPE(i)
} // namespace detail
} // namespace doctest
