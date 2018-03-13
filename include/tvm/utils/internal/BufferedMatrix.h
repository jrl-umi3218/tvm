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
  /** This class provides a matrix with resizable buffer, so that alloc
    */
  class BufferedMatrix
  {
  public:
    BufferedMatrix(Eigen::DenseIndex m, Eigen::DenseIndex n);

    Eigen::Map<const Eigen::MatrixXd, Eigen::Aligned> get() const;
    Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> get();

    void resize(Eigen::DenseIndex m, Eigen::DenseIndex n);

  private:
    Eigen::DenseIndex m_;
    Eigen::DenseIndex n_;
    Eigen::VectorXd buffer_;
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
