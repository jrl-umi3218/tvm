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

  VariableVector::VariableVector(const std::vector<VariablePtr>& variables)
    : VariableVector()
  {
    add(variables);
  }

  VariableVector::VariableVector(std::initializer_list<VariablePtr> variables)
    : VariableVector()
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

  void VariableVector::add(const VariableVector& variables, bool mergeDuplicate)
  {
    add(variables.variables(), mergeDuplicate);
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
    assert(val.size() == size());
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
