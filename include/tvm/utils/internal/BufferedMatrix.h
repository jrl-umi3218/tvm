/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <Eigen/Core>

namespace tvm
{

namespace utils
{

namespace internal
{
/** This class provides a matrix with resizable buffer, so that allocations
 * are made only if there is not enough space.
 *
 */
class BufferedMatrix
{
public:
  /** Build a m-by-n BufferedMatrix. A buffer twice as big as needed is
   * created.
   */
  BufferedMatrix(Eigen::DenseIndex m, Eigen::DenseIndex n);

  /** Get a map to the matrix. The matrix has the size specified at
   * construction or during the last resize.\
   */
  Eigen::Map<const Eigen::MatrixXd, Eigen::Aligned> get() const;
  /** Get a map to the matrix. The matrix has the size specified at
   * construction or during the last resize.
   */
  Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> get();

  /** Resize the matrix. If the buffer is not big enough, reallocate memory
   * to twice the needed size.
   */
  void resize(Eigen::DenseIndex m, Eigen::DenseIndex n);

  template<typename Derived>
  Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> & operator=(const Eigen::EigenBase<Derived> & xpr);

private:
  Eigen::DenseIndex m_;    /** Row size of the matrix*/
  Eigen::DenseIndex n_;    /** Column size of the matrix*/
  Eigen::VectorXd buffer_; /** vector, used as buffer*/
};

inline BufferedMatrix::BufferedMatrix(Eigen::DenseIndex m, Eigen::DenseIndex n) { resize(m, n); }

inline Eigen::Map<const Eigen::MatrixXd, Eigen::Aligned> BufferedMatrix::get() const
{
  return Eigen::Map<const Eigen::MatrixXd, Eigen::Aligned>(buffer_.data(), m_, n_);
}

inline Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> BufferedMatrix::get()
{
  return Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>(buffer_.data(), m_, n_);
}

inline void BufferedMatrix::resize(Eigen::DenseIndex m, Eigen::DenseIndex n)
{
  if(m * n > buffer_.size())
  {
    buffer_.resize(2 * m * n);
  }
  m_ = m;
  n_ = n;
}

template<typename Derived>
inline Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> & BufferedMatrix::operator=(const Eigen::EigenBase<Derived> & xpr)
{
  resize(xpr.rows(), xpr.cols());
  return get() = xpr;
}

} // namespace internal

} // namespace utils

} // namespace tvm
