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
    * \internal FIXME? Have a version which is compatible with Eigen::NoAlias (if needed).
    */
  template<typename MatrixType>
  class KeepProperties
  {
  public:
    /** Create the proxy.
      *
      * \param M The MatrixWithProperties
      * \param keep Specifies if the properties should be kept (\p true) or not
      * (\p false)
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

  /** An Eigen object together with MatrixProperties. */
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
