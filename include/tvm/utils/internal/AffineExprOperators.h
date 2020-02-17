/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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

#include <tvm/utils/AffineExpr.h>

namespace tvm
{

namespace utils
{

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
  return make_AffineExpr(-aff.constant(), tvm::utils::internal::tupleUnaryMinus(aff.linear(), std::make_index_sequence<sizeof...(Derived)>{}));
}

/** Aff = m*Aff, where \p m is any type accepted by Eigen for pre-multiplication 
  * (in particular scalar or matrix expression).
  */
template<typename MultDerived, typename CstDerived, typename... Derived>
inline auto operator*(const MultDerived& m, const tvm::utils::AffineExpr<CstDerived, Derived...>& aff)
{
  return make_AffineExpr(m * aff.constant(), tvm::utils::internal::tuplePremult(m, aff.linear(), std::make_index_sequence<sizeof...(Derived)>{}));
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
  return { lhs.constant() + rhs, lhs.linear() };
}

/** Aff = b + Aff */
template<typename LhsDerived, typename CstDerived, typename ... RhsDerived>
inline tvm::utils::AffineExpr<tvm::utils::internal::AddConstantsRetType<LhsDerived, CstDerived>, RhsDerived...>
operator+(const Eigen::MatrixBase<LhsDerived>& lhs, const tvm::utils::AffineExpr<CstDerived, RhsDerived...>& rhs)
{
  return { lhs + rhs.constant(), rhs.linear() };
}

/** Aff = Aff + Aff */
template<typename LCstDerived, typename RCstDerived, typename ... LhsDerived, typename ... RhsDerived>
inline tvm::utils::AffineExpr<tvm::utils::internal::AddConstantsRetType<LCstDerived, RCstDerived>, LhsDerived..., RhsDerived...>
operator+(const tvm::utils::AffineExpr<LCstDerived, LhsDerived...>& lhs, const tvm::utils::AffineExpr<RCstDerived, RhsDerived...>& rhs)
{
  return { lhs.constant() + rhs.constant(), std::tuple_cat(lhs.linear(), rhs.linear()) };
}

/** Lin = m * Lin */
template<typename MultType, typename Derived>
inline auto operator*(const MultType& m, const tvm::utils::LinearExpr<Derived>& lin)
{
  return tvm::utils::LinearExpr<std::remove_const_t<decltype(m * std::declval<Derived>())>>(m * lin.matrix(), lin.variable());
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

} // namespace utils

/** Lin = M * x */
template<typename Derived>
inline tvm::utils::LinearExpr<Derived>
operator*(const Eigen::MatrixBase<Derived>& matrix, const tvm::VariablePtr& v)
{
  return { matrix, v };
}

/** Lin = -var */
inline tvm::utils::LinearExpr<tvm::utils::internal::MinusIdentityType> operator-(const tvm::VariablePtr& v)
{
  return v;
}

/** Lin = scalar * var */
inline tvm::utils::LinearExpr<tvm::utils::internal::MultIdentityType> operator*(double s, const tvm::VariablePtr& v)
{
  assert(s != 0);
  return { s, v };
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

} // namespace tvm
