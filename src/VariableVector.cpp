/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/VariableVector.h>

#include <tvm/Variable.h>

namespace tvm
{
int VariableVector::counter = 0;

VariableVector::VariableVector() : size_(0) { getNewStamp(); }

VariableVector::VariableVector(const VariableVector & other) : VariableVector(other.variables()) {}

VariableVector & VariableVector::operator=(const VariableVector & other)
{
  clear();
  add(other.variables());
  return *this;
}

VariableVector::~VariableVector() { clear(); }

bool VariableVector::add(VariablePtr v)
{
  if((v->size() == 0) || contains(*v.get()))
  {
    return false;
  }
  for(const auto & xi : variables_)
  {
    if(xi->contains(*v))
    {
      return false;
    }
    else if(xi->intersects(*v))
    {
      throw std::runtime_error("[VariableVector::add] Attempting to add a variable that intersects a variable already "
                               "present but is not contained by it");
    }
  }
  add_(v);
  return true;
}

bool VariableVector::add(std::unique_ptr<Variable> v) { return add(VariablePtr(std::move(v))); }

void VariableVector::add(const std::vector<VariablePtr> & variables)
{
  for(auto & v : variables)
    add(v);
}

void VariableVector::add(const VariableVector & variables) { add(variables.variables()); }

int VariableVector::addAndGetIndex(VariablePtr v, bool containingIndex)
{
  auto it = find_if(variables_.begin(), variables_.end(),
                    [&v](const VariablePtr & it) { return it->contains(*v) || it->intersects(*v); });
  if(it == variables_.end())
  {
    add_(v);
    return static_cast<int>(variables_.size() - 1);
  }
  else
  {
    if(*(it->get()) == *v) // it is equal to v
    {
      return static_cast<int>(it - variables_.begin());
    }
    else if((*it)->contains(*v)) // v is strictly contained in it
    {
      if(containingIndex)
        return static_cast<int>(it - variables_.begin());
      else
        return -1;
    }
    else // v only intersects it
    {
      throw std::runtime_error("[VariableVector::add] Attempting to add a variable that intersects a variable already "
                               "present but is not contained by it");
    }
  }
}

bool VariableVector::remove(const Variable & v)
{
  auto it =
      std::find_if(variables_.begin(), variables_.end(), [&v](const VariablePtr & it) { return (*(it.get()) == v); });
  if(it == variables_.end())
  {
    return false;
  }
  else
  {
    remove_(it);
    return true;
  }
}

void VariableVector::remove(int i)
{
  if(i < 0 || i >= static_cast<int>(variables_.size()))
  {
    throw std::out_of_range("[VariableVector::remove] Invalid index.");
  }

  auto it = variables_.begin() + i;
  remove_(it);
}

void VariableVector::clear()
{
  for(const auto & v : variables_)
    v->startIn_.erase(id());
  variables_.clear();
  size_ = 0;
  getNewStamp();
}

int VariableVector::totalSize() const { return size_; }

int VariableVector::numberOfVariables() const { return static_cast<int>(variables_.size()); }

const VariablePtr VariableVector::operator[](int i) const
{
  assert(i >= 0 && i < numberOfVariables());
  return variables_[i];
}

const std::vector<VariablePtr> & VariableVector::variables() const { return variables_; }

const Eigen::VectorXd & VariableVector::value() const
{
  value_.resize(size_);
  int n = 0;
  for(const auto & v : variables_)
  {
    int s = v->size();
    value_.segment(n, s) = v->value();
    n += s;
  }
  return value_;
}

void VariableVector::set(const VectorConstRef & val)
{
  assert(val.size() == totalSize());
  int n = 0;
  for(const auto & v : variables_)
  {
    int s = v->size();
    v->set(val.segment(n, s));
    n += s;
  }
}

void VariableVector::setZero()
{
  for(auto & v : variables_)
  {
    v->setZero();
  }
}

bool VariableVector::contains(const Variable & v) const
{
  auto it = find_if(variables_.begin(), variables_.end(), [&v](const VariablePtr & it) { return it->contains(v); });
  return it != variables_.end();
}

bool VariableVector::intersects(const Variable & v) const
{
  auto it = find_if(variables_.begin(), variables_.end(), [&v](const VariablePtr & it) { return it->intersects(v); });
  return it != variables_.end();
}

int VariableVector::indexOf(const Variable & v) const
{
  auto it = find_if(variables_.begin(), variables_.end(), [&v](const VariablePtr & it) { return it->contains(v); });
  if(it == variables_.end() || (*(it->get()) != v))
  {
    return -1;
  }
  else
  {
    return static_cast<int>(it - variables_.begin());
  }
}

Range VariableVector::getMappingOf(const Variable & v) const
{
  for(const auto & xi : variables_)
  {
    if(xi->contains(v))
    {
      int start = xi->startIn_[id()].start + static_cast<int>(v.value_.data() - xi->value_.data());
      v.startIn_[id()] = {start, stamp()};
      return {start, v.size()};
    }
  }

  throw std::runtime_error("This variable is not part of the vector of variables.");
}

int VariableVector::stamp() const { return stamp_; }

void VariableVector::add_(VariablePtr v)
{
  variables_.push_back(v);
  v->startIn_[id()] = {size_, -1}; // Variables added directly to the vector do no need a stamp
  size_ += v->size();
  getNewStamp();
}

void VariableVector::remove_(std::vector<VariablePtr>::const_iterator it)
{
  int si = (*it)->size();
  for(auto iti = it + 1; iti != end(); ++iti)
    (*iti)->startIn_[id()].start -= si;
  size_ -= si;
  (*it)->startIn_.erase(id());
  variables_.erase(it);
  getNewStamp();
}

void VariableVector::getNewStamp() const
{
  stamp_ = counter;
  counter++;
}

VariableVector TVM_DLLAPI dot(const VariableVector & vars, int ndiff)
{
  VariableVector dv;
  const auto & vv = vars.variables();
  for(const auto & v : vv)
  {
    dv.add(dot(v, ndiff));
  }
  return dv;
}
} // namespace tvm
