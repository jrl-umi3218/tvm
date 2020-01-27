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
