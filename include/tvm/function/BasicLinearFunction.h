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


#include <tvm/function/abstract/LinearFunction.h>
#include <tvm/internal/MatrixProperties.h>

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
    void add(const Eigen::MatrixXd& A, VariablePtr x);
  };

}  // namespace function

}  // namespace tvm
