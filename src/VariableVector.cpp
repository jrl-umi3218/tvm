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

#include <tvm/VariableVector.h>

#include <tvm/Variable.h>

namespace tvm
{
  int VariableVector::counter = 0;

  VariableVector::VariableVector()
    : size_(0)
  {
    getNewStamp();
  }

  bool VariableVector::add(VariablePtr v)
  {
    if (contains(*v.get()))
    {
      return false;
    }
    add_(v);
    return true;
  }

  bool VariableVector::add(std::unique_ptr<Variable> v)
  {
    return add(VariablePtr(std::move(v)));
  }


  void VariableVector::add(const std::vector<VariablePtr>& variables)
  {
    for (auto& v : variables)
      add(v);
  }

  void VariableVector::add(const VariableVector& variables)
  {
    add(variables.variables());
  }

  int VariableVector::addAndGetIndex(VariablePtr v)
  {
    auto it = find_if(variables_.begin(), variables_.end(), [&v](const VariablePtr& it) {return (it.get() == v.get()); });
    if (it == variables_.end())
    {
      add_(v);
      return static_cast<int>(variables_.size() - 1);
    }
    else
    {
      return static_cast<int>(it - variables_.begin());
    }
  }

  bool VariableVector::remove(const Variable& v)
  {
    if (!contains(v))
    {
      return false;
    }
    auto it = std::find_if(variables_.begin(), variables_.end(), [&v](const VariablePtr& it) {return (it.get() == &v); });
    remove_(it);
    return true;
  }

  void VariableVector::remove(int i)
  {
    if (i < 0 || i >= static_cast<int>(variables_.size()))
    {
      throw std::out_of_range("[VariableVector::remove] Invalid index.");
    }

    auto it = variables_.begin() + i;
    remove_(it);
  }

  int VariableVector::totalSize() const
  {
    return size_;
  }

  int VariableVector::numberOfVariables() const
  {
    return static_cast<int>(variables_.size());
  }

  const VariablePtr VariableVector::operator[](int i) const
  {
    assert(i >= 0 && i < numberOfVariables());
    return variables_[i];
  }

  const std::vector<VariablePtr>& VariableVector::variables() const
  {
    return variables_;
  }

  const Eigen::VectorXd& VariableVector::value() const
  {
    value_.resize(size_);
    int n = 0;
    for (const auto& v : variables_)
    {
      int s = v->size();
      value_.segment(n, s) = v->value();
      n += s;
    }
    return value_;
  }

  void VariableVector::value(const VectorConstRef& val)
  {
    assert(val.size() == totalSize());
    int n = 0;
    for (const auto& v : variables_)
    {
      int s = v->size();
      v->value(val.segment(n, s));
      n += s;
    }
  }

  void VariableVector::computeMapping() const
  {
    int size = 0;
    for (const auto& v : variables_)
    {
      v->mappingHelper_.start = size;
      v->mappingHelper_.stamp = stamp_;
      size += v->size();
    }
  }

  std::map<const Variable*, Range> VariableVector::computeMappingMap() const
  {
    computeMapping();
    std::map<const Variable*, Range> m;
    for (const auto& v : variables_)
    {
      m[v.get()] = { v->mappingHelper_.start, v->size() };
    }

    return m;
  }

  bool VariableVector::contains(const Variable& v) const
  {
    auto it = find_if(variables_.begin(), variables_.end(), [&v](const VariablePtr& it) {return (it.get() == &v); });
    return it != variables_.end();
  }

  int VariableVector::indexOf(const Variable & v) const
  {
    auto it = find_if(variables_.begin(), variables_.end(), [&v](const VariablePtr& it) {return (it.get() == &v); });
    if (it == variables_.end())
    {
      return -1;
    }
    else
    {
      return static_cast<int>(it - variables_.begin());
    }
  }

  int VariableVector::stamp() const
  {
    return stamp_;
  }

  void VariableVector::add_(VariablePtr v)
  {
    variables_.push_back(v);
    size_ += v->size();
    getNewStamp();
  }

  void VariableVector::remove_(std::vector<VariablePtr>::const_iterator it)
  {
    size_ -= (*it)->size();
    variables_.erase(it);
    getNewStamp();
  }

  void VariableVector::getNewStamp() const
  {
    stamp_ = counter;
    counter++;
  }

  VariableVector TVM_DLLAPI dot(const VariableVector& vars, int ndiff)
  {
    VariableVector dv;
    const auto& vv = vars.variables();
    for (const auto& v : vv)
    {
      dv.add(dot(v, ndiff));
    }
    return dv;
  }
}  // namespace tvm
