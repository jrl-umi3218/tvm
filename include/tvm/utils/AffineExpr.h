/* Copyright 2017-2019 CNRS-AIST JRL and CNRS-UM LIRMM
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

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/internal/meta.h>
#include <tvm/utils/internal/AffineExprDetail.h>

#include <eigen/Core>

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
    LinearExpr(const Eigen::MatrixBase<Derived>& matrix, const VariablePtr& v) : matrix_(matrix.derived()), var_(v) { assert(matrix.cols() == v->size()); }

    /** Constructor for Identity linear expression */
    template <class T = Derived, typename std::enable_if_t<std::is_same_v<T, internal::IdentityType>, int> = 0>
    LinearExpr(const tvm::VariablePtr& v) : matrix_(Eigen::MatrixXd::Identity(v->size(), v->size())), var_(v) {}

    /** Constructor for a*v where a is a scalar */
    template <class T = Derived, typename std::enable_if_t<std::is_same_v<T, internal::MultIdentityType>, int> = 0>
    LinearExpr(double a, const tvm::VariablePtr& v) : matrix_(a* Eigen::MatrixXd::Identity(v->size(), v->size())), var_(v) {}

    /** Constructor for -v */
    template <class T = Derived, typename std::enable_if_t<std::is_same_v<T, internal::MinusIdentityType>, int> = 0>
    LinearExpr(const tvm::VariablePtr& v) : matrix_(-Eigen::MatrixXd::Identity(v->size(), v->size())), var_(v) {}


    const Derived& matrix() const { return matrix_; };
    const VariablePtr& variable() const { return var_; }
  private:
    /** The matrix expression. */
    typename internal::RefSelector_t<Derived> matrix_;
    /** The variable */
    const VariablePtr& var_;
  };


  /** A class representing an affine expression A1 * x1 + A2 *x2 + ... + Ak * xk + b
    * where Ai is a matrix expression, xi is a Variable and b is a vector expression.
    *
    * The class is essentially a list of LinearExpr in the form of a std:::tuple, and a vector expression.
    *
    * \tparam CstDerived The type of the constant part. Possibly NoConstant, otherwise an Eigen
    *         expression type with the characteristics of a vector.
    * \tparam Derived The list of matrix expression types corresponding to each Ai
    */
  template<typename CstDerived, typename ... Derived>
  class AffineExpr
  {
  public:
    /** Construction from a sequence of LinearExpr, in the case of CstDerived == NoConstant (absent constant part).*/
    template<class T = CstDerived, typename std::enable_if_t<std::is_same_v<T, internal::NoConstant>, int> = 0>
    AffineExpr(const LinearExpr<Derived>&... linear)
      : linear_(std::forward_as_tuple(linear ...)), constant_(internal::NoConstant())
    {}

    /** Construction from a sequence of LinearExpr, in the case of CstDerived != NoConstant (existing constant part).*/
    template <class T = CstDerived, typename std::enable_if_t<!std::is_same_v<T, internal::NoConstant>, int> = 0>
    AffineExpr(const Eigen::MatrixBase<CstDerived>& constant, const LinearExpr<Derived>&... linear)
      : linear_(std::forward_as_tuple(linear ...)), constant_(constant.derived())
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(CstDerived)
    }

    /** Construction from a tuple of LinearExpr, in the case of CstDerived == NoConstant (absent constant part).*/
    template <class T = CstDerived, typename std::enable_if_t<std::is_same_v<T, internal::NoConstant>, int> = 0>
    AffineExpr(const internal::NoConstant& constant, const std::tuple<LinearExpr<Derived>... >& linear)
      : linear_(linear), constant_(constant)
    {}

    /** Construction from a tuple of LinearExpr, in the case of CstDerived != NoConstant (existing constant part).*/
    template <class T = CstDerived, typename std::enable_if_t<!std::is_same_v<T, internal::NoConstant>, int> = 0>
    AffineExpr(const Eigen::MatrixBase<CstDerived>& constant, const std::tuple<LinearExpr<Derived>... >& linear)
      : linear_(linear), constant_(constant.derived())
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(CstDerived)
    }

    const std::tuple<LinearExpr<Derived>... >& linear() const { return linear_; }
    const CstDerived& constant() const { return constant_; }

  private:
    // ConstantType::Type is NoConstant in case CstType==NoConstant and ref_selector<CstDerived> otherwise.
    // Use of ref_selector<CstDerived>: for a vector, we need to keep a const ref, while for a vector
    // expression we need to take a copy of the expression (same use as in e.g. CWiseBinaryOp).
    template<typename T> struct ConstantType { using Type = typename internal::RefSelector_t<CstDerived>; };
    template<> struct ConstantType<internal::NoConstant> { using Type = internal::NoConstant; };

    /** The list of linear expressions Ai * xi */
    std::tuple<LinearExpr<Derived>... > linear_;
    /** The constant part b */
    typename ConstantType<CstDerived>::Type constant_;
  };

  /** Helper function to create a AffineExpr and infer automatically the template arguments.*/
  template<typename CstDerived, typename... Derived>
  AffineExpr<CstDerived, Derived...> make_AffineExpr(const CstDerived& constant, const std::tuple<LinearExpr<Derived>... >& linear)
  {
    return { constant, linear };
  }
}
}

/** Lin = M * x */
template<typename Derived>
inline tvm::utils::LinearExpr<Derived>
operator*(const Eigen::MatrixBase<Derived>& matrix, const tvm::VariablePtr& v)
{
  return { matrix, v };
}

/** Aff = Lin + b */
template<typename Derived, typename CstDerived>
inline tvm::utils::AffineExpr<CstDerived, Derived>
operator+(const tvm::utils::LinearExpr<Derived>& lin, const Eigen::MatrixBase<CstDerived>& cst)
{
  return { cst, lin };
}

/** Aff = b + Lin */
template<typename Derived, typename CstDerived>
inline tvm::utils::AffineExpr<CstDerived, Derived>
operator+(const Eigen::MatrixBase<CstDerived>& cst, const tvm::utils::LinearExpr<Derived>& lin)
{
  return { cst, lin };
}

/** Aff = Lin + Lin */
template<typename LhsDerived, typename RhsDerived>
inline tvm::utils::AffineExpr<tvm::utils::internal::NoConstant, LhsDerived, RhsDerived>
operator+(const tvm::utils::LinearExpr<LhsDerived>& lhs, const tvm::utils::LinearExpr<RhsDerived>& rhs)
{
  return { lhs, rhs };
}

/** Aff = Aff + Lin */
template<typename RhsDerived, typename CstDerived, typename ... LhsDerived>
inline tvm::utils::AffineExpr<CstDerived, LhsDerived..., RhsDerived>
operator+(const tvm::utils::AffineExpr<CstDerived, LhsDerived...>& lhs, const tvm::utils::LinearExpr<RhsDerived>& rhs)
{
  return { lhs.constant(), std::tuple_cat(lhs.linear(), std::forward_as_tuple(rhs)) };
}

/** Aff = Lin + Aff */
template<typename LhsDerived, typename CstDerived, typename ... RhsDerived>
inline tvm::utils::AffineExpr<CstDerived, LhsDerived, RhsDerived...>
operator+(const tvm::utils::LinearExpr<LhsDerived>& lhs, const tvm::utils::AffineExpr<CstDerived, RhsDerived...>& rhs)
{
  return { rhs.constant(), std::tuple_cat(std::forward_as_tuple(lhs), rhs.linear()) };
}

/** Aff = Aff + b */
template<typename RhsDerived, typename CstDerived, typename ... LhsDerived>
inline tvm::utils::AffineExpr<tvm::utils::internal::AddConstantsRetType<CstDerived, RhsDerived>, LhsDerived...>
operator+(const tvm::utils::AffineExpr<CstDerived, LhsDerived...>& lhs, const Eigen::MatrixBase<RhsDerived>& rhs)
{
  return { tvm::utils::internal::addConstants(lhs.constant(), rhs), lhs.linear() };
}

/** Aff = b + Aff */
template<typename LhsDerived, typename CstDerived, typename ... RhsDerived>
inline tvm::utils::AffineExpr<tvm::utils::internal::AddConstantsRetType<LhsDerived, CstDerived>, RhsDerived...>
operator+(const Eigen::MatrixBase<LhsDerived>& lhs, const tvm::utils::AffineExpr<CstDerived, RhsDerived...>& rhs)
{
  return { tvm::utils::internal::addConstants(lhs, rhs.constant()), rhs.linear() };
}

/** Aff = Aff + Aff */
template<typename LCstDerived, typename RCstDerived, typename ... LhsDerived, typename ... RhsDerived>
inline tvm::utils::AffineExpr<tvm::utils::internal::AddConstantsRetType<LCstDerived, RCstDerived>, LhsDerived..., RhsDerived...>
operator+(const tvm::utils::AffineExpr<LCstDerived, LhsDerived...>& lhs, const tvm::utils::AffineExpr<RCstDerived, RhsDerived...>& rhs)
{
  return { tvm::utils::internal::addConstants(lhs.constant(), rhs.constant()), std::tuple_cat(lhs.linear(), rhs.linear()) };
}

/** Lin = -var */
inline tvm::utils::LinearExpr<tvm::utils::internal::MinusIdentityType> operator-(const tvm::VariablePtr& v)
{
  return v;
}

/** Lin = -Lin */
template<typename Derived>
inline auto operator-(const tvm::utils::LinearExpr<Derived>& lin)
{
  return tvm::utils::LinearExpr<std::remove_const_t<decltype(-std::declval<Derived>())>>(-lin.matrix(), lin.variable());
}

/** Aff = -Aff */
template<typename CstDerived, typename... Derived>
inline auto operator-(const tvm::utils::AffineExpr<CstDerived, Derived...>& aff)
{
  return make_AffineExpr(tvm::utils::internal::cstUnaryMinus(aff.constant()), tvm::utils::internal::tupleUnaryMinus(aff.linear(), std::make_index_sequence<sizeof...(Derived)>{}));
}

/** Aff = m*Aff, where a is any type accepted by Eigen for pre-multiplication 
  * (in particular scalar or matrix expression.
  */
template<typename MultDerived, typename CstDerived, typename... Derived>
inline auto operator*(const MultDerived& m, const tvm::utils::AffineExpr<CstDerived, Derived...>& aff)
{
  return make_AffineExpr(tvm::utils::internal::CstMult<CstDerived>::run(m, aff.constant()), tvm::utils::internal::tuplePremult(m, aff.linear(), std::make_index_sequence<sizeof...(Derived)>{}));
}

/** Lin = scalar * var */
inline tvm::utils::LinearExpr<tvm::utils::internal::MultIdentityType> operator*(double s, const tvm::VariablePtr& v)
{
  assert(s != 0);
  return { s, v };
}

/** Lin = m * Lin */
template<typename MultType, typename Derived>
inline auto operator*(const MultType& m, const tvm::utils::LinearExpr<Derived>& lin)
{
  return tvm::utils::LinearExpr<std::remove_const_t<decltype(m * std::declval<Derived>())>>(m * lin.matrix(), lin.variable());
}

/** Aff = var + var */
inline auto operator+(const tvm::VariablePtr& u, const tvm::VariablePtr& v)
{
  return tvm::utils::LinearExpr<tvm::utils::internal::IdentityType>(u) 
    + tvm::utils::LinearExpr<tvm::utils::internal::IdentityType>(v);
}

/** Aff = a + var */
template<typename AddType, typename tvm::internal::enable_for_templated_t<AddType, Eigen::MatrixBase, tvm::utils::LinearExpr, tvm::utils::AffineExpr> = 0>
inline auto operator+(const AddType& a, const tvm::VariablePtr& v)
{
  return a + tvm::utils::LinearExpr<tvm::utils::internal::IdentityType>(v);
}

/** Aff = var + a */
template<typename AddType, typename tvm::internal::enable_for_templated_t<AddType, Eigen::MatrixBase, tvm::utils::LinearExpr, tvm::utils::AffineExpr> = 0>
inline auto operator+(const tvm::VariablePtr& v, const AddType& a)
{
  return tvm::utils::LinearExpr<tvm::utils::internal::IdentityType>(v) + a;
}

/** Aff = var - var */
inline auto operator-(const tvm::VariablePtr& u, const tvm::VariablePtr& v)
{
  return tvm::utils::LinearExpr<tvm::utils::internal::IdentityType>(u) + tvm::utils::LinearExpr<tvm::utils::internal::MinusIdentityType>(v);
}

/** Aff = vec - var */
template<typename SubType, typename tvm::internal::enable_for_templated_t<SubType, Eigen::MatrixBase, tvm::utils::LinearExpr, tvm::utils::AffineExpr> = 0>
inline auto operator-(const SubType& a, const tvm::VariablePtr& v)
{
  return a + tvm::utils::LinearExpr<tvm::utils::internal::MinusIdentityType>(v);
}

/** Aff = var - vec */
template<typename SubType, typename tvm::internal::enable_for_templated_t<SubType, Eigen::MatrixBase, tvm::utils::LinearExpr, tvm::utils::AffineExpr> = 0>
inline auto operator-(const tvm::VariablePtr& v, const SubType& a)
{
  return tvm::utils::LinearExpr<tvm::utils::internal::IdentityType>(v) + -a;
}

/** Aff = Lin - a */
template<typename Derived, typename SubType, typename tvm::internal::enable_for_templated_t<SubType, Eigen::MatrixBase, tvm::utils::LinearExpr, tvm::utils::AffineExpr> = 0>
inline auto operator-(const tvm::utils::LinearExpr<Derived>& lin, const SubType& rhs)
{
  return lin + -rhs;
}

/** Aff = Aff - a */
template<typename CstDerived, typename SubType, typename... Derived, typename tvm::internal::enable_for_templated_t<SubType, Eigen::MatrixBase, tvm::utils::LinearExpr, tvm::utils::AffineExpr> = 0>
inline auto operator-(const tvm::utils::AffineExpr<CstDerived, Derived...>& aff, const SubType& rhs)
{
  return aff + -rhs;
}