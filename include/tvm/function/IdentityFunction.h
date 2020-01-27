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

  protected:
    void updateValue_() override;
    void updateVelocity_() override;

  private:
    /** Overriden function that always throws.*/
    void A(const MatrixConstRef& A, const Variable& x,
      const tvm::internal::MatrixProperties& p = tvm::internal::MatrixProperties()) override;
    /** Overriden function that always throws.*/
    void A(const MatrixConstRef& A,
      const tvm::internal::MatrixProperties& p = tvm::internal::MatrixProperties()) override;
    /** Overriden function that always throws.*/
    void b(const VectorConstRef& b, const tvm::internal::MatrixProperties&) override;
  };

}  // namespace function

}  // namespace tvm
