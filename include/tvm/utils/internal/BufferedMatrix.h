#pragma once

/* Copyright 2017-2018 CNRS-UM LIRMM, CNRS-AIST JRL
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
      * to twice the neede size.
      */
    void resize(Eigen::DenseIndex m, Eigen::DenseIndex n);

  private:
    Eigen::DenseIndex m_;     /** Row size of the matrix*/
    Eigen::DenseIndex n_;     /** Column size of the matrix*/
    Eigen::VectorXd buffer_;  /** vector, used as buffer*/
  };

  inline BufferedMatrix::BufferedMatrix(Eigen::DenseIndex m, Eigen::DenseIndex n)
  {
    resize(m, n);
  }

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
    if (m*n > buffer_.size())
    {
      buffer_.resize(2 * m*n);
    }
    m_ = m;
    n_ = n;
  }


} //internal

} // utils

} //utils
