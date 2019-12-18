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


#include <tvm/function/abstract/LinearFunction.h>
#include <tvm/internal/MatrixProperties.h>
#include <tvm/utils/AffineExpr.h>

namespace tvm
{

namespace function
{

  /** The most basic linear function f(x_1, ..., x_k) = sum A_i x_i + b
   * where the matrices are constant.
   */
  class TVM_DLLAPI BasicLinearFunction : public abstract::LinearFunction
  {
  public:
    /** A x (b = 0) */
    BasicLinearFunction(const MatrixConstRef& A, VariablePtr x);
    /** A1 x1 + ... An xn (b = 0) */
    BasicLinearFunction(const std::vector<MatrixConstRef>& A, const std::vector<VariablePtr>& x);

    /** A x + b */
    BasicLinearFunction(const MatrixConstRef& A, VariablePtr x, const VectorConstRef& b);
    /** A1 x1 + ... An xn + b*/
    BasicLinearFunction(const std::vector<MatrixConstRef>& A, const std::vector<VariablePtr>& x, const VectorConstRef& b);

    /** Uninitialized version for a function of size \p m with a single variable
      * \p x
      * Don't forget to initialize A \b and b
      */
    BasicLinearFunction(int m, VariablePtr x);
    /** Uninitialized version for a function of size \p m with multiple
      * variables \p x1 ... \p xn
      * Don't forget to initialize the Ai \b and b
      */
    BasicLinearFunction(int m, const std::vector<VariablePtr>& x);

    template<typename Derived>
    BasicLinearFunction(const utils::LinearExpr<Derived>& lin);

    template<typename CstDerived, typename... Derived>
    BasicLinearFunction(const utils::AffineExpr<CstDerived, Derived...>& aff);

    /** Set the matrix \p A corresponding to variable \p x and optionally the
      * properties \p p of \p A.*/
    virtual void A(const MatrixConstRef& A, const Variable& x,
                   const internal::MatrixProperties& p = internal::MatrixProperties());
    /** Shortcut for when there is a single variable.*/
    virtual void A(const MatrixConstRef& A, 
                   const internal::MatrixProperties& p = internal::MatrixProperties());
    /** Set the constant term \p b, and optionally its properties \p p.*/
    virtual void b(const VectorConstRef& b, const internal::MatrixProperties& p = internal::MatrixProperties());

    using LinearFunction::b;

  private:
    template<typename Derived>
    void add(const Eigen::MatrixBase<Derived>& A, VariablePtr x);

    template<typename Tuple, size_t ... Indices>
    void add(const Tuple& tuple, std::index_sequence<Indices...>);

    template<typename Derived>
    void add(const utils::LinearExpr<Derived>& lin);
  };

  template<typename Derived>
  BasicLinearFunction::BasicLinearFunction(const utils::LinearExpr<Derived>& lin)
    : LinearFunction(static_cast<int>(lin.matrix().rows()))
  {
    add(lin);
    this->b(Eigen::VectorXd::Zero(size()),
            { tvm::internal::MatrixProperties::Constness(true),
              tvm::internal::MatrixProperties::ZERO });
    setDerivativesToZero();
  }

  template<typename CstDerived, typename... Derived>
  BasicLinearFunction::BasicLinearFunction(const utils::AffineExpr<CstDerived, Derived...>& aff)
    : LinearFunction(static_cast<int>(std::get<0>(aff.linear()).matrix().rows()))
  {
    constexpr int N = std::tuple_size_v< std::tuple<utils::LinearExpr<Derived>... > >;
    add(aff.linear(), std::make_index_sequence<N>{});
    if constexpr (std::is_same_v<CstDerived, utils::internal::NoConstant>)
    {
      this->b(Eigen::VectorXd::Zero(size()),
              { tvm::internal::MatrixProperties::Constness(true),
                tvm::internal::MatrixProperties::ZERO });
    }
    else
    {
      this->b(aff.constant(), { tvm::internal::MatrixProperties::Constness(true) });
    }
    setDerivativesToZero();
  }

  template<typename Derived>
  inline void BasicLinearFunction::add(const Eigen::MatrixBase<Derived>& A, VariablePtr x)
  {
    if (!x->space().isEuclidean() && x->isBasePrimitive())
      throw std::runtime_error("We allow linear function only on Euclidean variables.");
    if (A.rows() != size())
      throw std::runtime_error("Matrix A doesn't have coherent row size.");
    if (A.cols() != x->size())
      throw std::runtime_error("Matrix A doesn't have its column size coherent with its corresponding variable.");
    
    if (variables().contains(*x))
    {
      jacobian_.at(x.get()).noalias() += A;
    }
    else
    {
      addVariable(x, true);
      jacobian_.at(x.get()) = A;
      jacobian_.at(x.get()).properties({ tvm::internal::MatrixProperties::Constness(true) });
    }
  }

  template<typename Tuple, size_t ... Indices>
  inline void BasicLinearFunction::add(const Tuple& tuple, std::index_sequence<Indices...>)
  {
    // trick to call add on each element indexed by Indices:
    // - ( add(std::get<I>(tuple)), 42 ) for any integer I returns 42 (second element of (X, 42), 
    //   X is evaluated but is then ignored).
    // - We create an initializer_list by pack expansion where each expanded element is a pair as
    //   above. This evaluates add(std::get<I>(tuple)) for each I in Indices in turn.
    auto l = { (add(std::get<Indices>(tuple)), 42)... };
  }

  template<typename Derived>
  inline void BasicLinearFunction::add(const utils::LinearExpr<Derived>& lin)
  {
    add(lin.matrix(), lin.variable());
  }


}  // namespace function

}  // namespace tvm
