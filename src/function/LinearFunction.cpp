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

#include <tvm/function/abstract/LinearFunction.h>

#include <tvm/Variable.h>

namespace tvm
{

namespace function
{

namespace abstract
{

  LinearFunction::LinearFunction(int m)
    :Function(m), b_(m)
  {
    registerUpdates(Update::Value, &LinearFunction::updateValue);
    registerUpdates(Update::Velocity, &LinearFunction::updateVelocity);
    addInputDependency<LinearFunction>(Update::Value, *this, Output::Jacobian);
    addInputDependency<LinearFunction>(Update::Value, *this, Output::B);
    addInputDependency<LinearFunction>(Update::Velocity, *this, Output::Jacobian);
    addOutputDependency<LinearFunction>(Output::Value, Update::Value);
    addOutputDependency<LinearFunction>(Output::Velocity, Update::Velocity);
  }

  void LinearFunction::updateValue()
  {
    updateValue_();
  }

  void LinearFunction::updateVelocity()
  {
    updateVelocity_();
  }

  void LinearFunction::resizeCache()
  {
    Function::resizeCache();
    b_.resize(size());
    setDerivativesToZero();
  }

  void LinearFunction::updateValue_()
  {
    value_ = b_;
    for (auto v : variables())
      value_ += jacobian(*v) * v->value();
  }

  void LinearFunction::updateVelocity_()
  {
    velocity_.setZero();
    for (auto v : variables())
      velocity_ += jacobian(*v) * dot(v)->value();
  }

  void LinearFunction::setDerivativesToZero()
  {
    normalAcceleration_.setZero();
    for (const auto& v : variables())
      JDot_[v.get()].setZero();
  }

}  // namespace abstract

}  // namespace function

}  // namespace tvm
