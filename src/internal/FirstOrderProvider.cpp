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

#include <tvm/internal/FirstOrderProvider.h>

#include <tvm/exception/exceptions.h>

namespace tvm
{

namespace internal
{
  FirstOrderProvider::FirstOrderProvider(int m)
    : m_(m)
  {
    resizeCache(); //resize value_
  }

  void FirstOrderProvider::resizeCache()
  {
    resizeValueCache();
    resizeJacobianCache();
  }

  void FirstOrderProvider::resizeValueCache()
  {
    if (isOutputEnabled((int)Output::Value))
      value_.resize(m_);
  }

  void FirstOrderProvider::resizeJacobianCache()
  {
    if (isOutputEnabled((int)Output::Jacobian))
    {
      for (auto v : variables_.variables())
        jacobian_[v.get()].resize(m_, v->space().tSize());
    }
  }

  void FirstOrderProvider::addVariable(VariablePtr v, bool linear)
  {
    if(variables_.add(v))
    {
      jacobian_[v.get()].resize(m_, v->space().tSize());
      linear_[v.get()] = linear;

      addVariable_(v);
    }
  }

  void FirstOrderProvider::addVariable(const VariableVector & vv, bool linear)
  {
    for(auto v : vv.variables())
    {
      addVariable(v, linear);
    }
  }

  void FirstOrderProvider::removeVariable(VariablePtr v)
  {
    variables_.remove(*v);
    jacobian_.erase(v.get());
    removeVariable_(v);
  }

  void FirstOrderProvider::addVariable_(VariablePtr)
  {
    //do nothing
  }

  void FirstOrderProvider::removeVariable_(VariablePtr)
  {
    //do nothing
  }

  void FirstOrderProvider::splitJacobian(const MatrixConstRef & J, const std::vector<VariablePtr>& vars, bool keepProperties)
  {
    Eigen::DenseIndex s = 0;
    for (const auto& v : vars)
    {
      auto n = static_cast<Eigen::DenseIndex>(v->space().tSize());
      jacobian_[v.get()].keepProperties(keepProperties) = J.middleCols(s, n);
      s += n;
    }
  }

  void FirstOrderProvider::resize(int m)
  {
    m_ = m;
    resizeCache();
  }

}  // namespace internal

}  // namespace tvm
