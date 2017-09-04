#include "Variable.h"
#include "VariableVector.h"

namespace tvm
{
  int VariableVector::counter = 0;

  VariableVector::VariableVector()
  {
  }

  VariableVector::VariableVector(const std::vector<VariablePtr>& variables)
  {
    add(variables);
  }

  VariableVector::VariableVector(std::initializer_list<VariablePtr> variables)
  {
    for (auto& v : variables)
      add(v);
  }

  void VariableVector::add(VariablePtr v, bool mergeDuplicate)
  {
    if (contains(*v.get()))
    {
      if (mergeDuplicate)
        return;
      else
        throw std::runtime_error("Attempting to add a variable already present.");
    }
    variables_.push_back(v);
    variableSet_.insert(v.get());
    size_ += v->size();
    getNewStamp();
  }

  void VariableVector::add(const std::vector<VariablePtr>& variables, bool mergeDuplicate)
  {
    for (auto& v : variables)
      add(v, mergeDuplicate);
  }

  void VariableVector::remove(const Variable& v, bool ignoreAbsence)
  {
    if (contains(v))
    {
      auto it = std::find_if(variables_.begin(), variables_.end(), [&v](const VariablePtr& it) {return (it.get() == &v); });
      variables_.erase(it);
      variableSet_.erase(&v);
      size_ -= v.size();
      getNewStamp();
    }
    else
    {
      if (!ignoreAbsence)
        throw std::runtime_error("Attempting to remove a variable that is not present.");
    }
  }

  int VariableVector::size() const
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

  void VariableVector::computeMapping() const
  {
    getNewStamp();
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
    return variableSet_.find(&v) != variableSet_.end();
  }

  int VariableVector::stamp() const
  {
    return stamp_;
  }

  void VariableVector::getNewStamp() const
  {
    stamp_ = counter;
    counter++;
  }
}