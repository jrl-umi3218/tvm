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


#include <tvm/function/BasicLinearFunction.h>

namespace tvm
{

namespace function
{

  /** f(x) = x, for a single variable.*/
  class TVM_DLLAPI IdentityFunction : public BasicLinearFunction
  {
  public:
    /** Build an identity function on variable \p x*/
    IdentityFunction(VariablePtr x);

    /** Overriden function that always throws.*/
    void A(const MatrixConstRef& A, const Variable& x, 
           const tvm::internal::MatrixProperties& p = tvm::internal::MatrixProperties()) override;
    /** Overriden function that always throws.*/
    void A(const MatrixConstRef& A,
           const tvm::internal::MatrixProperties& p = tvm::internal::MatrixProperties()) override;
    /** Overriden function that always throws.*/
    void b(const VectorConstRef& b, const tvm::internal::MatrixProperties&) override;

  protected:
    void updateValue_() override;
    void updateVelocity_() override;
  };

}  // namespace function

}  // namespace tvm
