/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Variable.h>
#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/internal/meta.h>
#include <tvm/utils/internal/AffineExprDetail.h>

#include <Eigen/Core>

#include <tuple>

namespace tvm
{
namespace utils
{
/** A class representing an expression A * x with A a matrix expression and x a variable
 *
 * \tparam Derived The type of the matrix expression.
 */
template<typename Derived>
class LinearExpr
{
public:
  /** General constructor */
  LinearExpr(const Eigen::MatrixBase<Derived> & matrix, const VariablePtr & v) : matrix_(matrix.derived()), var_(v)
  {
    assert(matrix.cols() == v->size());
  }

  /** Constructor for Identity linear expression */
  template<class T = Derived, typename std::enable_if_t<std::is_same_v<T, internal::IdentityType>, int> = 0>
  LinearExpr(const tvm::VariablePtr & v) : matrix_(Eigen::MatrixXd::Identity(v->size(), v->size())), var_(v)
  {}

  /** Constructor for a*v where a is a scalar */
  template<class T = Derived, typename std::enable_if_t<std::is_same_v<T, internal::MultIdentityType>, int> = 0>
  LinearExpr(double a, const tvm::VariablePtr & v)
  : matrix_(a * Eigen::MatrixXd::Identity(v->size(), v->size())), var_(v)
  {}

  /** Constructor for -v */
  template<class T = Derived, typename std::enable_if_t<std::is_same_v<T, internal::MinusIdentityType>, int> = 0>
  LinearExpr(const tvm::VariablePtr & v) : matrix_(-Eigen::MatrixXd::Identity(v->size(), v->size())), var_(v)
  {}

  const Derived & matrix() const { return matrix_; };
  const VariablePtr & variable() const { return var_; }

private:
  /** The matrix expression. */
  typename internal::RefSelector_t<Derived> matrix_;
  /** The variable */
  const VariablePtr & var_;
};

/** A class representing an affine expression \f$ A_1 x_1 + A_2 x_2 + \ldots + A_k x_k + b \f$
 * where \f$ A_i \f$ is a matrix expression, \f$ x_i \f$ is a Variable and
 * \f$ b \f$ is a vector expression.
 *
 * The class is essentially a list of LinearExpr in the form of a std:::tuple,
 * and a vector expression.
 *
 * \tparam CstDerived The type of the constant part. Possibly NoConstant, otherwise an Eigen
 *         expression type with the characteristics of a vector.
 * \tparam Derived The list of matrix expression types corresponding to each Ai
 */
template<typename CstDerived, typename... Derived>
class AffineExpr
{
public:
  /** Construction from a sequence of LinearExpr, in the case of CstDerived == NoConstant (absent constant part).*/
  template<class T = CstDerived, typename std::enable_if_t<std::is_same_v<T, internal::NoConstant>, int> = 0>
  AffineExpr(const LinearExpr<Derived> &... linear)
  : linear_(std::forward_as_tuple(linear...)), constant_(internal::NoConstant())
  {}

  /** Construction from a sequence of LinearExpr, in the case of CstDerived != NoConstant (existing constant part).*/
  template<class T = CstDerived, typename std::enable_if_t<!std::is_same_v<T, internal::NoConstant>, int> = 0>
  AffineExpr(const Eigen::MatrixBase<CstDerived> & constant, const LinearExpr<Derived> &... linear)
  : linear_(std::forward_as_tuple(linear...)), constant_(constant.derived())
  {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(CstDerived)
  }

  /** Construction from a tuple of LinearExpr, in the case of CstDerived == NoConstant (absent constant part).*/
  template<class T = CstDerived, typename std::enable_if_t<std::is_same_v<T, internal::NoConstant>, int> = 0>
  AffineExpr(const internal::NoConstant & constant, const std::tuple<LinearExpr<Derived>...> & linear)
  : linear_(linear), constant_(constant)
  {}

  /** Construction from a tuple of LinearExpr, in the case of CstDerived != NoConstant (existing constant part).*/
  template<class T = CstDerived, typename std::enable_if_t<!std::is_same_v<T, internal::NoConstant>, int> = 0>
  AffineExpr(const Eigen::MatrixBase<CstDerived> & constant, const std::tuple<LinearExpr<Derived>...> & linear)
  : linear_(linear), constant_(constant.derived())
  {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(CstDerived)
  }

  const std::tuple<LinearExpr<Derived>...> & linear() const { return linear_; }
  const CstDerived & constant() const { return constant_; }

private:
  // ConstantType::Type is NoConstant in case CstType==NoConstant and ref_selector<CstDerived> otherwise.
  // Use of ref_selector<CstDerived>: for a vector, we need to keep a const ref, while for a vector
  // expression we need to take a copy of the expression (same use as in e.g. CWiseBinaryOp).
  template<typename T>
  struct ConstantType
  {
    using Type = typename internal::RefSelector_t<CstDerived>;
  };

  /** The list of linear expressions Ai * xi */
  std::tuple<LinearExpr<Derived>...> linear_;
  /** The constant part b */
  typename ConstantType<CstDerived>::Type constant_;
};

/** Helper function to create a AffineExpr and infer automatically the template arguments.*/
template<typename CstDerived, typename... Derived>
AffineExpr<CstDerived, Derived...> make_AffineExpr(const CstDerived & constant,
                                                   const std::tuple<LinearExpr<Derived>...> & linear)
{
  return {constant, linear};
}
} // namespace utils
} // namespace tvm

#include <tvm/utils/internal/AffineExprOperators.h>
