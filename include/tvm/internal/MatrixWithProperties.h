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
  //forward declaration
  class MatrixWithProperties;

  /** A lightweight proxy for indicating if the assignement of an Eigen
    * expression to a MatrixWithProperties should reset the properties (which is
    * the default behavior.
    *
    * FIXME? Have a version which is compatible with Eigen::NoAlias (if needed).
    */
  class KeepProperties
  {
  public:
    /** Create the proxy.
      *
      * \param M The MatrixWithProperties
      * \param keep Specified is the properties should be kept or not
      */
    KeepProperties(MatrixWithProperties& M, bool keep) : M_(M), keep_(keep) {}

    /** The assignment operator for Eigen's expression*/
    template<typename OtherDerived>
    MatrixWithProperties& operator=(const Eigen::MatrixBase<OtherDerived>& other);

    /** We delete this operator so that it cannot be used for the classical copy
      * operator of MatrixWithProperties. */
    MatrixWithProperties& operator=(const MatrixWithProperties&) = delete;

  private:
    MatrixWithProperties& M_;
    const bool keep_;
  };

  /** An Eigen matrix together with MatrixProperties. */
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

    template<typename OtherDerived>
    MatrixWithProperties& assignKeepProperties(const Eigen::MatrixBase<OtherDerived>& other)
    {
      assert(this->rows() == other.rows() && this->cols() == other.cols()
        && "It is not allowed to assign an expression with a different size. Please explicitely resize the matrix before.");
      this->Eigen::MatrixXd::operator=(other);
      return *this;
    }

    const MatrixProperties& properties() const { return properties_; }
    void properties(const MatrixProperties& p) { properties_ = p; }

    /** Create a proxy to specfy wether an assignement should preserve the
      * properties of the matrix. 
      *
      * \param keep true it the properties should be left untouched, false
      * otherwise.
      */
    KeepProperties keepProperties(bool keep) { return { *this, keep }; }

  private:
    MatrixProperties properties_;
  };

  template<typename OtherDerived>
  inline MatrixWithProperties& KeepProperties::operator=(const Eigen::MatrixBase<OtherDerived>& other)
  {
    if (keep_)
    {
      M_.assignKeepProperties(other);
    }
    else
    {
      M_ = other;
    }
    return M_;
  }

}  // namespace internal

}  // namespace tvm
