/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/diagnostic/GraphProbe.h>

namespace tvm::diagnostic
{
GraphProbe::GraphProbe(const graph::internal::Log & log) : log_(log) {}

std::vector<GraphProbe::OutputVal> GraphProbe::listOutputVal(bool verbose) const
{
  std::vector<OutputVal> ret;

  for(auto o : log_.outputs_)
  {
    if(auto it = outputAccessor_.find({o.owner.type.hash_code(), o.id}); it != outputAccessor_.end())
      ret.push_back({o, nullptr, it->second(o.owner.value)});

    if(auto it = varDepOutputAccessor_.find({o.owner.type.hash_code(), o.id}); it != varDepOutputAccessor_.end())
    {
      const auto & varMatPair = it->second(o.owner.value);
      for(const auto & p : varMatPair)
        ret.push_back({o, p.first, p.second});
    }
  }
  return ret;
}
} // namespace tvm::diagnostic