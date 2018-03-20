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
  template<typename MatrixType>
  class ObjectWithProperties;

  /** A lightweight proxy for indicating if the assignement of an Eigen
    * expression to a MatrixWithProperties should reset the properties (which is
    * the default behavior.
    *
    * FIXME? Have a version which is compatible with Eigen::NoAlias (if needed).
    */
  template<typename MatrixType>
  class KeepProperties
  {
  public:
    /** Create the proxy.
      *
      * \param M The MatrixWithProperties
      * \param keep Specified is the properties should be kept or not
      */
    KeepProperties(ObjectWithProperties<MatrixType>& M, bool keep) : M_(M), keep_(keep) {}

    /** The assignment operator for Eigen's expression*/
    template<typename OtherDerived>
    ObjectWithProperties<MatrixType>& operator=(const Eigen::MatrixBase<OtherDerived>& other);

    /** We delete this operator so that it cannot be used for the classical copy
      * operator of MatrixWithProperties. */
    ObjectWithProperties<MatrixType>& operator=(const ObjectWithProperties<MatrixType>&) = delete;

  private:
    ObjectWithProperties<MatrixType>& M_;
    const bool keep_;
  };

  /** An Eigen matrix together with MatrixProperties. */
  template<typename MatrixType>
  class ObjectWithProperties : public MatrixType
  {
  public:
    using MatrixType::MatrixType;

    ObjectWithProperties() {}

    template<typename OtherDerived>
    ObjectWithProperties(const Eigen::MatrixBase<OtherDerived>& other, const MatrixProperties& p)
      : MatrixType(other), properties_(p)
    {
    }

    template<typename OtherDerived>
    ObjectWithProperties& operator=(const Eigen::MatrixBase<OtherDerived>& other)
    {
      assert(this->rows() == other.rows() && this->cols() == other.cols()
        && "It is not allowed to assign an expression with a different size. Please explicitely resize the matrix before.");
      this->MatrixType::operator=(other);
      properties_ = MatrixProperties();
      return *this;
    }

    template<typename OtherDerived>
    ObjectWithProperties& assignKeepProperties(const Eigen::MatrixBase<OtherDerived>& other)
    {
      assert(this->rows() == other.rows() && this->cols() == other.cols()
        && "It is not allowed to assign an expression with a different size. Please explicitely resize the matrix before.");
      this->MatrixType::operator=(other);
      return *this;
    }

    const MatrixProperties& properties() const { return properties_; }
    void properties(const MatrixProperties& p) { properties_ = p; }

    /** Create a proxy to specify wether an assignement should preserve the
      * properties of the matrix. 
      *
      * \param keep true it the properties should be left untouched, false
      * otherwise.
      */
    KeepProperties<MatrixType> keepProperties(bool keep) { return { *this, keep }; }

  private:
    MatrixProperties properties_;
  };

  template<typename MatrixType>
  template<typename OtherDerived>
  inline ObjectWithProperties<MatrixType>& KeepProperties<MatrixType>::operator=(const Eigen::MatrixBase<OtherDerived>& other)
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

  using MatrixWithProperties = ObjectWithProperties<Eigen::MatrixXd>;
  using VectorWithProperties = ObjectWithProperties<Eigen::VectorXd>;

}  // namespace internal

}  // namespace tvm
