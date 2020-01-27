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
