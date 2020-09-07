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

#include <tvm/function/BasicLinearFunction.h>

#include <tvm/Variable.h>

namespace tvm
{

namespace function
{

BasicLinearFunction::BasicLinearFunction(const MatrixConstRef & A, VariablePtr x)
: BasicLinearFunction({A}, std::vector<VariablePtr>{x})
{
}

BasicLinearFunction::BasicLinearFunction(const std::vector<MatrixConstRef> & A, const std::vector<VariablePtr> & x)
: BasicLinearFunction(A, x, Eigen::VectorXd::Zero(A.begin()->rows()))
{
  this->b_.properties({tvm::internal::MatrixProperties::Constness(true), tvm::internal::MatrixProperties::ZERO});
}

BasicLinearFunction::BasicLinearFunction(const MatrixConstRef & A, VariablePtr x, const VectorConstRef & b)
: BasicLinearFunction({A}, std::vector<VariablePtr>{x}, b)
{
}

BasicLinearFunction::BasicLinearFunction(const std::vector<MatrixConstRef> & A,
                                         const std::vector<VariablePtr> & x,
                                         const VectorConstRef & b)
: LinearFunction(static_cast<int>(A.begin()->rows()))
{
  if(A.size() != x.size())
    throw std::runtime_error("The number of matrices and variables is incoherent.");

  auto v = x.begin();
  for(const Eigen::MatrixXd & a : A)
  {
    add(a, *v);
    ++v;
  }
  this->b(b);
  setDerivativesToZero();
}

BasicLinearFunction::BasicLinearFunction(int m, VariablePtr x) : BasicLinearFunction(m, std::vector<VariablePtr>{x}) {}

BasicLinearFunction::BasicLinearFunction(int m, const std::vector<VariablePtr> & x) : LinearFunction(m)
{
  for(auto & v : x)
  {
    addVariable(v, true);
    jacobian_.at(v.get()).properties({tvm::internal::MatrixProperties::Constness(true)});
  }
  setDerivativesToZero();
}

void BasicLinearFunction::A(const MatrixConstRef & A, const Variable & x, const tvm::internal::MatrixProperties & p)
{
  if(A.rows() == size() && A.cols() == x.size())
  {
    jacobian_.at(&x) = A;
    jacobian_.at(&x).properties(p);
  }
  else
    throw std::runtime_error("Matrix A doesn't have the good size.");
}

void BasicLinearFunction::A(const MatrixConstRef & A, const tvm::internal::MatrixProperties & p)
{
  if(variables().numberOfVariables() == 1)
    this->A(A, *variables()[0].get(), p);
  else
    throw std::runtime_error("You can use this method only for constraints with one variable.");
}

void BasicLinearFunction::b(const VectorConstRef & b, const internal::MatrixProperties & p)
{
  if(b.size() == size())
  {
    this->b_ = b;
    this->b_.properties(p);
  }
  else
    throw std::runtime_error("Vector b doesn't have the correct size.");
}

} // namespace function

} // namespace tvm
