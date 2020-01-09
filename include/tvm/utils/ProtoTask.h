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

#include <tvm/api.h>
#include <tvm/utils/AffineExpr.h>
#include <tvm/utils/internal/ProtoTaskDetails.h>

namespace tvm
{

namespace utils
{

  template<constraint::Type T, typename FunT>
  class ProtoTaskCommon;

  /** A utiliy class to represent the "constraint" part of a Task, for general functions*/
  template<constraint::Type T>
  using ProtoTask = ProtoTaskCommon<T, FunctionPtr>;

  /** A utiliy class to represent the "constraint" part of a Task, specialized for linear functions*/
  template<constraint::Type T>
  using LinearProtoTask = ProtoTaskCommon<T, LinearFunctionPtr>;

  template<constraint::Type T, typename FunT>
  class ProtoTaskCommon
  {
  public:
    ProtoTaskCommon(FunT f, const internal::RHS& rhs)
    : f_(f), rhs_(rhs)
    {
      if (rhs.type_ == internal::RHSType::Vector && f->size() != rhs.v_.size())
      {
        throw std::runtime_error("The vector you provided has not the correct size.");
      }
    }

    template<typename FunU>
    ProtoTaskCommon(const ProtoTaskCommon<T, FunU> & pt)
    : f_(pt.f_), rhs_(pt.rhs_)
    {
    }

    FunT f_;
    internal::RHS rhs_;
  };

  template<typename FunT>
  class ProtoTaskCommon<constraint::Type::DOUBLE_SIDED, FunT>
  {
  public:
    ProtoTaskCommon(FunT f, const internal::RHS& l, const internal::RHS & u)
    : f_(f), l_(l), u_(u)
    {
      if (l.type_ == internal::RHSType::Vector && f->size() != l.v_.size())
      {
        throw std::runtime_error("The lower bound vector you provided has not the correct size.");
      }
      if (u.type_ == internal::RHSType::Vector && f->size() != u.v_.size())
      {
        throw std::runtime_error("The upper bound vector you provided has not the correct size.");
      }
    }

    template<typename FunU>
    ProtoTaskCommon(const ProtoTaskCommon<constraint::Type::DOUBLE_SIDED, FunU> & pt)
    : f_(pt.f_), l_(pt.l_), u_(pt.u_)
    {
    }

    FunctionPtr f_;
    internal::RHS l_;
    internal::RHS u_;
  };

  /** A helper alias that is IfNotLinearFunction if T is a tvm::abstract::Function, 
    * but not a tvm::abstract::LinearFunction, IfLinearFunction if it is a 
    * tvm::abstract::LinearFunction, and nothing (discarded by SFINAE) otherwise.
    */
  template<typename T, typename IfNotLinearFunction, typename IfLinearFunction>
  using ProtoChoice = typename std::enable_if<
                                 std::is_base_of<tvm::function::abstract::Function, T>::value, 
                                 typename std::conditional<
                                   std::is_base_of<tvm::function::abstract::LinearFunction, T>::value, 
                                   IfLinearFunction, 
                                   IfNotLinearFunction
                                 >::type
                               >::type;

  /** Equality ProtoTask f = rhs*/
  using ProtoTaskEQ = ProtoTask<constraint::Type::EQUAL>;
  using LinearProtoTaskEQ = LinearProtoTask<constraint::Type::EQUAL>;
  template<typename T>
  using ProtoTaskEQRet = ProtoChoice<T, tvm::utils::ProtoTaskEQ, tvm::utils::LinearProtoTaskEQ>;

  /** Inequality ProtoTask f <= rhs*/
  using ProtoTaskLT = ProtoTask<constraint::Type::LOWER_THAN>;
  using LinearProtoTaskLT = LinearProtoTask<constraint::Type::LOWER_THAN>;
  template<typename T>
  using ProtoTaskLTRet = ProtoChoice<T, tvm::utils::ProtoTaskLT, tvm::utils::LinearProtoTaskLT>;

  /** Inequality ProtoTask f >= rhs*/
  using ProtoTaskGT = ProtoTask<constraint::Type::GREATER_THAN>;
  using LinearProtoTaskGT = LinearProtoTask<constraint::Type::GREATER_THAN>;
  template<typename T>
  using ProtoTaskGTRet = ProtoChoice<T, tvm::utils::ProtoTaskGT, tvm::utils::LinearProtoTaskGT>;

  /** Double sided inequality ProtoTask l <= f <= u*/
  using ProtoTaskDS = ProtoTask<constraint::Type::DOUBLE_SIDED>;
  using LinearProtoTaskDS = LinearProtoTask<constraint::Type::DOUBLE_SIDED>;
  template<typename T>
  using ProtoTaskDSRet = ProtoChoice<T, tvm::utils::ProtoTaskDS, tvm::utils::LinearProtoTaskDS>;

} // namespace utils

} // namespace tvm

/** Conveniency operators to form a ProtoTask or LinearProtoTask f op rhs
  * (or l <= f <= u)
  *
  * \param f the function to form the task
  * \param rhs a double or a Eigen::Vector with the sane size as the function.
  * Note that for a double you need to explicitely write a double (e.g 0.,
  * not 0), otherwise the compiler won't be able to decide wich overload to
  * pick between this and shared_ptr operator.
  */
///@{
template<typename F>
inline tvm::utils::ProtoTaskEQRet<F> operator==(std::shared_ptr<F> f, const tvm::utils::internal::RHS& rhs) { return { f, rhs }; }
template<typename F>
inline tvm::utils::ProtoTaskEQRet<F> operator==(const tvm::utils::internal::RHS& rhs, std::shared_ptr<F> f) { return { f, rhs }; }
template<typename F>
inline tvm::utils::ProtoTaskGTRet<F> operator>=(std::shared_ptr<F> f, const tvm::utils::internal::RHS& rhs) { return { f, rhs }; }
template<typename F>
inline tvm::utils::ProtoTaskLTRet<F> operator>=(const tvm::utils::internal::RHS& rhs, std::shared_ptr<F> f) { return { f, rhs }; }
template<typename F>
inline tvm::utils::ProtoTaskLTRet<F> operator<=(std::shared_ptr<F> f, const tvm::utils::internal::RHS& rhs) { return { f, rhs }; }
template<typename F>
inline tvm::utils::ProtoTaskGTRet<F> operator<=(const tvm::utils::internal::RHS& rhs, std::shared_ptr<F> f) { return { f, rhs }; }

inline tvm::utils::ProtoTaskDS operator>=(const tvm::utils::ProtoTaskLT& ptl, const tvm::utils::internal::RHS& rhs) { return { ptl.f_, rhs, ptl.rhs_ }; }
inline tvm::utils::ProtoTaskDS operator<=(const tvm::utils::ProtoTaskGT& ptg, const tvm::utils::internal::RHS& rhs) { return { ptg.f_, ptg.rhs_, rhs }; }
inline tvm::utils::LinearProtoTaskDS operator>=(const tvm::utils::LinearProtoTaskLT& ptl, const tvm::utils::internal::RHS& rhs) { return { ptl.f_, rhs, ptl.rhs_ }; }
inline tvm::utils::LinearProtoTaskDS operator<=(const tvm::utils::LinearProtoTaskGT& ptg, const tvm::utils::internal::RHS& rhs) { return { ptg.f_, ptg.rhs_, rhs }; }

#define TVM_ID(x) std::make_shared<tvm::function::IdentityFunction>(x)

inline tvm::utils::LinearProtoTaskEQ operator==(tvm::VariablePtr x, const tvm::utils::internal::RHS& rhs) { return TVM_ID(x) == rhs; }
inline tvm::utils::LinearProtoTaskEQ operator==(const tvm::utils::internal::RHS& rhs, tvm::VariablePtr x) { return TVM_ID(x) == rhs; }
inline tvm::utils::LinearProtoTaskGT operator>=(tvm::VariablePtr x, const tvm::utils::internal::RHS& rhs) { return TVM_ID(x) >= rhs; }
inline tvm::utils::LinearProtoTaskLT operator>=(const tvm::utils::internal::RHS& rhs, tvm::VariablePtr x) { return TVM_ID(x) <= rhs; }
inline tvm::utils::LinearProtoTaskLT operator<=(tvm::VariablePtr x, const tvm::utils::internal::RHS& rhs) { return TVM_ID(x) <= rhs; }
inline tvm::utils::LinearProtoTaskGT operator<=(const tvm::utils::internal::RHS& rhs, tvm::VariablePtr x) { return TVM_ID(x) >= rhs; }
///@}
#undef TVM_ID

#define TVM_LIN(x) std::make_shared<tvm::function::BasicLinearFunction>(x)

/** Conveniency operators to form a LinearProtoTask expr op rhs (or l <= expr <= u)
  * where expr is a linear expression of the form matrixExpr * VariablePtr or an
  * affine expression as a sum of linear expressions and vectorExpr.
  *
  * \param lin the linear expression to form the task
  * \param aff the affine expression to form the task
  * \param rhs a double or a Eigen::Vector with the sane size as the function.
  * Note that for a double you don't need to explicitely write a double (e.g 0.)
  * to the contrary of the operators working with shared_ptr on Function.
  */
///@{
template<typename Derived>
inline tvm::utils::LinearProtoTaskEQ operator==(const tvm::utils::LinearExpr<Derived>& lin, const tvm::utils::internal::RHS& rhs) { return TVM_LIN(lin) == rhs; }
template<typename Derived>
inline tvm::utils::LinearProtoTaskEQ operator==(const tvm::utils::internal::RHS& rhs, const tvm::utils::LinearExpr<Derived>& lin) { return TVM_LIN(lin) == rhs; }
template<typename Derived>
inline tvm::utils::LinearProtoTaskGT operator>=(const tvm::utils::LinearExpr<Derived>& lin, const tvm::utils::internal::RHS& rhs) { return TVM_LIN(lin) >= rhs; }
template<typename Derived>
inline tvm::utils::LinearProtoTaskLT operator>=(const tvm::utils::internal::RHS& rhs, const tvm::utils::LinearExpr<Derived>& lin) { return TVM_LIN(lin) <= rhs; }
template<typename Derived>
inline tvm::utils::LinearProtoTaskLT operator<=(const tvm::utils::LinearExpr<Derived>& lin, const tvm::utils::internal::RHS& rhs) { return TVM_LIN(lin) <= rhs; }
template<typename Derived>
inline tvm::utils::LinearProtoTaskGT operator<=(const tvm::utils::internal::RHS& rhs, const tvm::utils::LinearExpr<Derived>& lin) { return TVM_LIN(lin) >= rhs; }

template<typename CstDerived, typename... Derived>
inline tvm::utils::LinearProtoTaskEQ operator==(const tvm::utils::AffineExpr<CstDerived, Derived...>& aff, const tvm::utils::internal::RHS& rhs) { return TVM_LIN(aff) == rhs; }
template<typename CstDerived, typename... Derived>
inline tvm::utils::LinearProtoTaskEQ operator==(const tvm::utils::internal::RHS& rhs, const tvm::utils::AffineExpr<CstDerived, Derived...>& aff) { return TVM_LIN(aff) == rhs; }
template<typename CstDerived, typename... Derived>
inline tvm::utils::LinearProtoTaskGT operator>=(const tvm::utils::AffineExpr<CstDerived, Derived...>& aff, const tvm::utils::internal::RHS& rhs) { return TVM_LIN(aff) >= rhs; }
template<typename CstDerived, typename... Derived>
inline tvm::utils::LinearProtoTaskLT operator>=(const tvm::utils::internal::RHS& rhs, const tvm::utils::AffineExpr<CstDerived, Derived...>& aff) { return TVM_LIN(aff) <= rhs; }
template<typename CstDerived, typename... Derived>
inline tvm::utils::LinearProtoTaskLT operator<=(const tvm::utils::AffineExpr<CstDerived, Derived...>& aff, const tvm::utils::internal::RHS& rhs) { return TVM_LIN(aff) <= rhs; }
template<typename CstDerived, typename... Derived>
inline tvm::utils::LinearProtoTaskGT operator<=(const tvm::utils::internal::RHS& rhs, const tvm::utils::AffineExpr<CstDerived, Derived...>& aff) { return TVM_LIN(aff) >= rhs; }
///@}
#undef TVM_LIN