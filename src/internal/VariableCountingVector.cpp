/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/internal/VariableCountingVector.h>

#include <tvm/Variable.h>

#include <sstream>

namespace tvm::internal
{
bool VariableCountingVector::add(VariablePtr v)
{
  VariablePtr s = v->superVariable();
  auto & counterPair = count_.insert({s.get(), {RangeCounting{}, 0}}).first->second;
  bool change = counterPair.first.add(v->subvariableRange());
  if(change)
    ++counterPair.second;
  upToDate_ = upToDate_ && !change;
  return change;
}

void VariableCountingVector::add(const VariableVector & v)
{
  for(const auto & x : v)
    add(x);
}

bool VariableCountingVector::remove(const Variable & v)
{
  VariablePtr s = v.superVariable();
  auto & counterPair = count_.at(s.get());
  bool change = counterPair.first.remove(v.subvariableRange());
  if(change)
  {
    if(counterPair.first.empty())
    {
      counterPair.second = 0;
    }
    else
    {
      // Removals that incur a change are penalized so that simple() will be false
      // for the corresponding variable
      counterPair.second += 10;
    }
  }
  upToDate_ = upToDate_ && !change;
  return change;
}

void VariableCountingVector::remove(const VariableVector & v)
{
  for(const auto & x : v)
    remove(*x);
}

void VariableCountingVector::clear()
{
  upToDate_ = false;
  count_.clear();
}

void VariableCountingVector::value(const VectorConstRef & val)
{
  assert(upToDate_);
  variables_.value(val);
}

const VariableVector & VariableCountingVector::variables() const
{
  update();
  return variables_;
}

const std::vector<uint8_t> VariableCountingVector::simple() const
{
  if(split_)
    throw std::runtime_error("[VariableCountingVector::simple] Only meaningful for split = false.");
  update();
  return simple_;
}
void VariableCountingVector::update() const
{
  if(!upToDate_)
  {
    variables_.clear();
    simple_.clear();
    for(const auto & p : count_)
    {
      auto ranges = p.second.first.ranges(split_);
      if(ranges.size() == 0)
      {
        continue;
      }
      else
      {
        auto v = p.first;
        bool simple = p.second.second == ranges.size();
        if(ranges.size() == 1 && ranges[0].dim == v->size())
        {
          assert(ranges[0].start == 0);
          variables_.add(v->shared_from_this());
          simple_.push_back(simple);
        }
        else
        {
          for(const auto & r : ranges)
          {
            std::stringstream ss;
            ss << v->name() << "[" << r.start << ":" << r.end() - 1 << "]";
            // The following line make the assumption that all variables can be treated as euclidean:
            // the dimension and the shift of the subvariable are defined with simple spaces.
            // This is fine if the VariableVector is not differentiated afterwards.
            // If this was really necessary to derive the vector, one could keep track of all the
            // space dimensions with several RangeCounting (one per dimension).
            variables_.add(v->subvariable(Space(r.dim), ss.str(), Space(r.start)));
            simple_.push_back(simple);
          }
        }
      }
    }
    upToDate_ = true;
  }
}
} // namespace tvm::internal
