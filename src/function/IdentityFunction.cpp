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

#include <tvm/function/IdentityFunction.h>

#include <tvm/Variable.h>

namespace tvm
{

namespace function
{

IdentityFunction::IdentityFunction(VariablePtr x)
  : BasicLinearFunction(Eigen::MatrixXd::Identity(x->size(), x->size()), x)
{
  jacobian_.begin()->second.properties({ tvm::internal::MatrixProperties::Shape::IDENTITY });
}

void IdentityFunction::A(const MatrixConstRef&, const Variable&, 
                         const tvm::internal::MatrixProperties&)
{
  throw std::runtime_error("You can not change the A matrix on a identity function");
}

void IdentityFunction::A(const MatrixConstRef&, const tvm::internal::MatrixProperties&)
{
  throw std::runtime_error("You can not change the A matrix on a identity function");
}

void IdentityFunction::b(const VectorConstRef&, const tvm::internal::MatrixProperties&)
{
  throw std::runtime_error("You can not change the b vector on a identity function");
}

void IdentityFunction::updateValue_()
{
  value_ = variables()[0]->value();
}

void IdentityFunction::updateVelocity_()
{
  velocity_ = dot(variables()[0])->value();
}


}  // namespace function

}  // namespace tvm
