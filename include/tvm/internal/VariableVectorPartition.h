/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Variable.h>
#include <tvm/internal/VariableCountingVector.h>

namespace tvm::internal
{
/** A helper class to iterate over a container of VariablePtr \a c where the
 * variables are split according to a given partition \a p expressed as a
 * VariableCountingVector. The iteration is made over \a c, but the iterator
 * points to (sub)variables in \a p.
 *
 * For example, let's consider x, y, z, 3 variables with size 8 and x1, x2, x3
 * (resp. y1, y2, y3 and z1, z2, z3) such that x = [x1,x2,x3] (same for y and z).
 * If u is a subvariable of x such that u = x2, v = y and w is a subvariable of z
 * such that w = [z2, z3], then iterating over VariableVectorPartition({w,u,v},
 * {x1, x2, x3, y1, y2, y3, z1, z2, z3}) will yield the sequence
 * {z2, z3, x2, y1, y2, y3}.
 *
 *       <- u->      <------- v------->      <---- w---->
 * <-x1-><-x2-><-x3-><-y1-><-y2-><-y3-><-z1-><-z2-><-z3->
 *
 * This is particularly useful when \a p is the result of the union of several
 * collections of variables with intersecting subvariables, leading to a
 * partition of the union with finer grain that any of the original collections.
 * Then we can iterate over any of these collections using this class and get
 * variables that will be either fully contained by or disjoint from any
 * variable of any collection.
 * As a very simplistic example, let's have x = [x1,x2,x3] and u and v two
 * subvariables of x such that u=[x1,x2] and v=[x2,x3]. Let c1 be a collection
 * containing u and c2 a collection containing v, and p the VariableCountingVector
 * obtained by adding u and v. p is equivalent to [x1,x2,x3]
 * Iterating over c1 with respect to p yields x1 then x2. Each of this variable
 * is clearly either in or out of c2, and we can rely on that to perform
 * computations. Likewise, iterating on c2 yields x2 then x3, which can be
 * compared to c1.
 *
 * The partition and container need to be compatible, i.e. any variable of the
 * container can be written as a union of variables in the partition.
 * This requirement is not fully checked.
 *
 * \tparam VarVector A container of VariablePtr with begin(), end() and forward iterator.
 *
 * \internal The partition relies on a VariableCountingVector which ensures that
 * consecutive subvariables in a variable are consecutive in the
 * VariableCountingVector::variables.
 * The VariableCountingVector needs to have the split option on, so that the
 * required compatibility is not with the variables added to it, but with what is
 * returned by VariableCountingVector::variables.
 */
template<class VarVector>
class VariableVectorPartition
{
public:
  /**
   * \param v The variable container we want to iterate over.
   * \param partition The way to split the variables. Need to have the split option on.
   * The default use should be to put all relevant variables in \p partition,
   * including all the variables in v, to make sure the partition is granularity
   * is small enough and compatible with v.
   */
  VariableVectorPartition(const VarVector & v, const VariableCountingVector & partition)
  : var_(v), partition_(partition.variables())
  {
    assert(partition.split());
  }

  class iterator
  {
  public:
    iterator(const VarVector & var, const VariableVector & p, int iv) : ip_(0), iv_(iv), var_(var), partition_(p)
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
