/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <iostream>
#include <tvm/internal/VariableCountingVector.h>

#include <tvm/Variable.h>

#include <sstream>

namespace tvm::internal
{
bool VariableCountingVector::add(VariablePtr v)
{
  VariablePtr s = v->superVariable();
  const Space & start = v->spaceShift();
  const Space & dim = v->space();
  auto & counterPair = count_.insert({s, {SpaceRangeCounting{}, 0}}).first->second;
  // bool change = counterPair.first.add(v->subvariableRange());
  bool change = counterPair.first.add(start, dim);
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
  std::cout << "VariableCountingVector removeVariable: " << v.name() << std::endl;
  VariablePtr s = v.superVariable();
  const Space & start = v.spaceShift();
  const Space & dim = v.space();
  auto & counterPair = count_.at(s);
  bool change = counterPair.first.remove(start, dim);
  if(change)
  {
    std::cout << "removed " << v.name() << std::endl;
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

void VariableCountingVector::set(const VectorConstRef & val)
{
  assert(upToDate_);
  variables_.set(val);
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

bool VariableCountingVector::isDisjointUnion()
{
  for(const auto & p : count_)
  {
    if(p.second.first.maxCount() > 1)
      return false;
  }
  return true;
}

void VariableCountingVector::update() const
{
  if(!upToDate_)
  {
    variables_.clear();
    simple_.clear();
    for(const auto & p : count_)
    {
      auto mRanges = p.second.first.mSize_.ranges(split_);
      auto rRanges = p.second.first.rSize_.ranges(split_);
      auto tRanges = p.second.first.tSize_.ranges(split_);
      assert(mRanges.size() == rRanges.size() && mRanges.size() == tRanges.size());
      if(mRanges.size() == 0)
      {
        continue;
      }
      else
      {
        auto v = p.first;
        bool simple = p.second.second == mRanges.size();
        if(mRanges.size() == 1
           && ((v->isBasePrimitive() && rRanges[0].dim == v->size())
               || (!v->isBasePrimitive() && tRanges[0].dim == v->size())))
        {
          assert(mRanges[0].start == 0);
          variables_.add(v);
          simple_.push_back(simple);
        }
        else
        {
          for(size_t i = 0; i < mRanges.size(); ++i)
          {
            const auto & mr = mRanges[i];
            const auto & rr = rRanges[i];
            const auto & tr = tRanges[i];
            auto sub = v->subvariable(Space(mr.dim, rr.dim, tr.dim), Space(mr.start, rr.start, tr.start));
            variables_.add(sub);
            simple_.push_back(simple);
          }
        }
      }
    }
    upToDate_ = true;
  }
}
bool VariableCountingVector::SpaceRangeCounting::add(const Space & start, const Space & dim)
{
  rSize_.add({start.rSize(), dim.rSize()});
  tSize_.add({start.tSize(), dim.tSize()});
  return mSize_.add({start.size(), dim.size()});
}

bool VariableCountingVector::SpaceRangeCounting::remove(const Space & start, const Space & dim)
{
  rSize_.remove({start.rSize(), dim.rSize()});
  tSize_.remove({start.tSize(), dim.tSize()});
  return mSize_.remove({start.size(), dim.size()});
}
} // namespace tvm::internal
