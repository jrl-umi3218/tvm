/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/internal/MatrixProperties.h>

#include <Eigen/Core>

namespace tvm
{

namespace internal
{
// forward declaration
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
  KeepProperties(ObjectWithProperties<MatrixType> & M, bool keep) : M_(M), keep_(keep) {}

  /** The assignment operator for Eigen's expression*/
  template<typename OtherDerived>
  ObjectWithProperties<MatrixType> & operator=(const Eigen::MatrixBase<OtherDerived> & other);

  /** We delete this operator so that it cannot be used for the classical copy
   * operator of MatrixWithProperties. */
  ObjectWithProperties<MatrixType> & operator=(const ObjectWithProperties<MatrixType> &) = delete;

private:
  ObjectWithProperties<MatrixType> & M_;
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
  ObjectWithProperties(const Eigen::MatrixBase<OtherDerived> & other)
  : MatrixType(other), properties_()
  {}

  template<typename OtherDerived>
  ObjectWithProperties(const Eigen::MatrixBase<OtherDerived> & other, const MatrixProperties & p)
  : MatrixType(other), properties_(p)
  {}

  template<typename OtherType>
  ObjectWithProperties(const ObjectWithProperties<OtherType> & other)
  : MatrixType(other), properties_(other.properties())
  {}

  template<typename OtherDerived>
  ObjectWithProperties & operator=(const Eigen::MatrixBase<OtherDerived> & other)
  {
    assert(this->rows() == other.rows() && this->cols() == other.cols()
           && "It is not allowed to assign an expression with a different size. Please explicitely resize the matrix "
              "before.");
    this->MatrixType::operator=(other);
    properties_ = MatrixProperties();
    return *this;
  }

  template<typename OtherDerived>
  ObjectWithProperties & assignKeepProperties(const Eigen::MatrixBase<OtherDerived> & other)
  {
    assert(this->rows() == other.rows() && this->cols() == other.cols()
           && "It is not allowed to assign an expression with a different size. Please explicitely resize the matrix "
              "before.");
    this->MatrixType::operator=(other);
    return *this;
  }

  const MatrixProperties & properties() const { return properties_; }
  void properties(const MatrixProperties & p) { properties_ = p; }

  /** Create a proxy to specify wether an assignement should preserve the
   * properties of the matrix.
   *
   * \param keep true it the properties should be left untouched, false
   * otherwise.
   */
  KeepProperties<MatrixType> keepProperties(bool keep) { return {*this, keep}; }

private:
  MatrixProperties properties_;
};

template<typename MatrixType>
template<typename OtherDerived>
inline ObjectWithProperties<MatrixType> & KeepProperties<MatrixType>::operator=(
    const Eigen::MatrixBase<OtherDerived> & other)
{
  if(keep_)
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
using MatrixConstRefWithProperties = ObjectWithProperties<Eigen::Ref<const Eigen::MatrixXd>>;

} // namespace internal

} // namespace tvm
