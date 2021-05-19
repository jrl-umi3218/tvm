/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/internal/meta.h>

#include <Eigen/Core>

namespace tvm::diagnostic
{
/** Check if a given address belongs to the data of a matrix
 * \param ptr The address to test.
 * \param M the matrix.
 *
 * \tparam MatrixOrMapType An Eigen type with a \c data method
 */
template<typename MatrixOrMapType>
inline bool isInMatrix(typename MatrixOrMapType::Scalar const * ptr, MatrixOrMapType & M)
{
  if constexpr(MatrixOrMapType::IsRowMajor)
  {
    static_assert(::tvm::internal::always_false<MatrixOrMapType>::value, "Only implemented for column-major matrices");
  }
  else
  {
    auto d = ptr - M.data();
    if(d < 0)
      return false;

    // within the column number
    if(d >= M.cols() * M.colStride())
      return false;

    // position in column
    auto r = (d % M.colStride());
    if(r >= M.rows() * M.rowStride())
      return false;

    return (r % M.rowStride() == 0);
  }
}

/** Check if a \c A(i,j) is an element of \p M
 *
 * \tparam MatrixOrMapType An Eigen type with a \c data method
 */
template<typename Derived, typename MatrixOrMapType>
inline bool isInMatrix(const Eigen::DenseCoeffsBase<Derived> & A, Eigen::Index i, Eigen::Index j, MatrixOrMapType & M)
{
  return isInMatrix(&A.coeff(i, j), M);
}

/** Check if any element \a e of \p A is such that \c emin<=abs(e)<=emax.*/
template<typename Derived>
inline bool hasElemInRange(const Eigen::MatrixBase<Derived> & A,
                           typename Derived::Scalar emin,
                           typename Derived::Scalar emax)
{
  return (A.array().abs() >= emin && A.array().abs() <= emax).any();
}

} // namespace tvm::diagnostic