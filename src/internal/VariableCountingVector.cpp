/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/internal/VariableCountingVector.h>

#include <tvm/Variable.h>

#include <sstream>

namespace tvm::internal
{
bool VariableCountingVector::add(VariablePtr v)
{
  VariablePtr s = v->superVariable();
  bool change = count_[s.get()].add(v->subvariableRange());
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
  bool change = count_.at(s.get()).remove(v.subvariableRange());
  upToDate_ = upToDate_ && !change;
  return change;
}

void VariableCountingVector::remove(const VariableVector & v)
{
  for(const auto & x : v)
    remove(*x);
}

void VariableCountingVector::value(const VectorConstRef & val) { variables_.value(val); }

const VariableVector & VariableCountingVector::variables() const
{
  if(!upToDate_)
  {
    variables_.clear();
    for(const auto & p : count_)
    {
      auto ranges = p.second.ranges();
      if(ranges.size() == 0)
      {
        continue;
      }
      else
      {
        auto v = p.first;
        if(ranges.size() == 1 && ranges[0].dim == v->size())
        {
          assert(ranges[0].start == 0);
          variables_.add(v->shared_from_this());
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
          }
        }
      }
    }
    upToDate_ = true;
  }
  return variables_;
}
} // namespace tvm::internal