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
    /** b = 0 */
    BasicLinearFunction(const MatrixConstRef& A, VariablePtr x);
    BasicLinearFunction(const std::vector<MatrixConstRef>& A, const std::vector<VariablePtr>& x);

    /** b is user-supplied*/
    BasicLinearFunction(const MatrixConstRef& A, VariablePtr x, const VectorConstRef& b);
    BasicLinearFunction(const std::vector<MatrixConstRef>& A, const std::vector<VariablePtr>& x, const VectorConstRef& b);

    /** Uninitialized version for a function of size \p m*/
    BasicLinearFunction(int m, VariablePtr x);
    BasicLinearFunction(int m, const std::vector<VariablePtr>& x);

    /** Set the matrix A corresponding to variable x.*/
    virtual void A(const MatrixConstRef& A, const Variable& x,
                   const tvm::internal::MatrixProperties& p = tvm::internal::MatrixProperties());
    /** Shortcut for when there is a single variable.*/
    virtual void A(const MatrixConstRef& A, 
                   const tvm::internal::MatrixProperties& p = tvm::internal::MatrixProperties());
    /** Set the constant term b.*/
    virtual void b(const VectorConstRef& b);

    using LinearFunction::b;

  private:
    void add(const Eigen::MatrixXd& A, VariablePtr x);
  };

}  // namespace function

}  // namespace tvm
