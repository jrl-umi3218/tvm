/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Variable.h>
#include <tvm/internal/VariableCountingVector.h>

namespace tvm::internal
{
template<class VarVector>
class VariableVectorPartition
{
public:
  VariableVectorPartition(const VarVector & v, const VariableCountingVector & partition)
  : var_(v), partition_(partition.variables())
  {
    assert(partition.split());
  }

  class iterator
  {
  public:
    iterator(const VarVector & var, const VariableVector & p, int iv) : var_(var), partition_(p), iv_(iv), ip_(0)
    {
      if(iv == p.numberOfVariables()) // end
      {
        ip_ = iv;
      }
      else
      {
        const auto & v = var[iv];
        for(; ip_ < p.numberOfVariables(); ++ip_)
        {
          if(v->contains(*p[ip_]))
            return;
        }
        throw std::runtime_error("Invalid partition");
      }
    }

    iterator & operator++()
    {
      ++ip_;
      if(ip_ < partition_.numberOfVariables() && var_[iv_]->contains(*partition_[ip_]))
      {
        return *this;
      }
      else
      {
        assert(partition_[ip_ - 1]->subvariableRange().end() == var_[iv_]->subvariableRange().end()
               && "End of variable in partition is not the same as the end of variable in the reference vector.");
        ++iv_;
        if(iv_ == static_cast<int>(var_.end() - var_.begin()))
        {
          // return end
          ip_ = partition_.numberOfVariables();
          return *this;
        }
        const auto & v = var_[iv_];
        for(ip_ = 0; ip_ < partition_.numberOfVariables(); ++ip_)
        {
          if(v->contains(*partition_[ip_]))
            return *this;
        }
        throw std::runtime_error("Invalid partition");
      }
    }

    bool operator==(iterator other) const
    {
      assert(&var_ == &other.var_ && &partition_ == &other.partition_);
      return ip_ == other.ip_;
    }
    bool operator!=(iterator other) const { return !(*this == other); }
    VariablePtr operator*() { return partition_[ip_]; }

  private:
    int ip_;
    int iv_;
    const VarVector & var_;
    const VariableVector & partition_;
  };

  iterator begin() { return {var_, partition_, 0}; }
  iterator end() { return {var_, partition_, partition_.numberOfVariables()}; }

private:
  const VarVector & var_;
  const VariableVector & partition_;
};
} // namespace tvm::internal