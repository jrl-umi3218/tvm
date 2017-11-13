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

#include <tvm/internal/MatrixProperties.h>

#include <Eigen/Core>

namespace tvm
{

namespace internal
{

  class MatrixWithProperties : public Eigen::MatrixXd
  {
  public:
    using Eigen::MatrixXd::MatrixXd;
    template<typename OtherDerived>
    MatrixWithProperties& operator=(const Eigen::MatrixBase<OtherDerived>& other)
    {
      assert(this->rows() == other.rows() && this->cols() == other.cols()
        && "It is not allowed to assign an expression with a different size. Please explicitely resize the matrix before.");
      this->Eigen::MatrixXd::operator=(other);
      properties_ = MatrixProperties();
      return *this;
    }
    const MatrixProperties& properties() const { return properties_; }
    void properties(MatrixProperties p) { properties_ = p; }

  private:
    MatrixProperties properties_;
  };

}  // namespace internal

}  // namespace tvm
