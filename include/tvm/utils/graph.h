/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/graph/CallGraph.h>
#include <tvm/graph/internal/Inputs.h>
#include <tvm/utils/internal/graphDetails.h>

#include <memory>

namespace tvm
{

namespace utils
{

/** Generate the graph of updates for a list of objects and their specified outputs
 *
 * g = generateUpdateGraph(obj1, o1_1, ..., o1_n1, obj2, o2_1, ..., o2_n2, ..., objk, ok_1, ..., ok_nk)
 * generates a graph g such that g.execute() ensures that outputs o1_1, ... o1_n1
 * for obj1, o2_1, ... o2_1 for obj2, etc. are up to date.
 */
template<typename Object, typename... Args>
inline std::unique_ptr<graph::CallGraph> generateUpdateGraph(std::shared_ptr<Object> obj, Args &&... args);

template<typename Object, typename... Args>
inline std::unique_ptr<graph::CallGraph> generateUpdateGraph(std::shared_ptr<Object> obj, Args &&... args)
{
  auto ptr = std::unique_ptr<graph::CallGraph>(new graph::CallGraph());
  auto user = std::make_shared<graph::internal::Inputs>();
  internal::parseSourcesAndOutputs(ptr.get(), user, obj, std::forward<Args>(args)...);
  ptr->update();

  return ptr;
}

} // namespace utils

} // namespace tvm
