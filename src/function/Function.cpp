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

#include <tvm/function/abstract/Function.h>

#include <tvm/Variable.h>

namespace tvm
{

namespace function
{

namespace abstract
{

  Function::Function(int m)
    : FirstOrderProvider(m)
  {
    resizeVelocityCache();
    resizeNormalAccelerationCache();
    resizeJDotCache();
  }

  void Function::resizeCache()
  {
    FirstOrderProvider::resizeCache();
    resizeVelocityCache();
    resizeNormalAccelerationCache();
    resizeJDotCache();
  }

  void Function::resizeVelocityCache()
  {
    if (isOutputEnabled((int)Output::Velocity))
      velocity_.resize(size());
  }

  void Function::resizeNormalAccelerationCache()
  {
    if (isOutputEnabled((int)Output::NormalAcceleration))
      normalAcceleration_.resize(size());
  }

  void Function::resizeJDotCache()
  {
    if (isOutputEnabled((int)Output::JDot))
    {
      for (auto v : variables())
        JDot_[v.get()].resize(size(), v->space().tSize());
    }
  }

  void Function::addVariable_(VariablePtr v)
  {
    JDot_[v.get()].resize(size(), v->space().tSize());
    variablesDot_.push_back(dot(v));
  }

  void Function::removeVariable_(VariablePtr v)
  {
    JDot_.erase(v.get());
    auto it = std::find(variablesDot_.begin(), variablesDot_.end(), dot(v));
    assert(it != variablesDot_.end()
      && "This should not happen: FirstOrderProvider::removeVariable would raise an exception first if the variable was not there.");
    variablesDot_.erase(it);
  }

}  // namespace abstract

}  // namespace function

}  // namespace tvm
