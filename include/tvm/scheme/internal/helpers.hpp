#pragma once

/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

namespace tvm
{
namespace scheme
{
namespace internal
{
class ProblemComputationData;

template<typename Problem, typename Scheme>
inline ProblemComputationData * getComputationData(Problem & problem, const Scheme & resolutionScheme)
{
  auto id = resolutionScheme.id();
  auto it = problem.computationData_.find(id);
  if(it != problem.computationData_.end())
  {
    return it->second.get();
  }
  else
  {
    problem.finalize();
    auto p =
        problem.computationData_.insert(std::move(std::make_pair(id, resolutionScheme.createComputationData(problem))));
    return p.first->second.get();
  }
}

} // namespace internal
} // namespace scheme
} // namespace tvm
