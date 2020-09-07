/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/graph/CallGraph.h>
#include <tvm/graph/internal/Inputs.h>

#include <memory>

namespace tvm
{

namespace utils
{

namespace internal
{
/** Recursion over the sources and outputs.
 * Case where the next element to parse is an output.
 */
template<typename Object, typename Output, typename... Args>
inline void parseSourcesAndOutputs(graph::CallGraph * g,
                                   std::shared_ptr<graph::internal::Inputs> user,
                                   std::shared_ptr<Object> obj,
                                   Output output,
                                   Args &&... args);

/** Recursion over the sources and outputs.
 * Case where the next element to parse is a source.
 */
template<typename Object1, typename Object2, typename... Args>
inline void parseSourcesAndOutputs(graph::CallGraph * g,
                                   std::shared_ptr<graph::internal::Inputs> user,
                                   std::shared_ptr<Object1> obj1,
                                   std::shared_ptr<Object2> obj2,
                                   Args &&... args);

/** Recursion over the sources and outputs.
 * End of recursion.
 */
template<typename Object>
inline void parseSourcesAndOutputs(graph::CallGraph * g,
                                   std::shared_ptr<graph::internal::Inputs> user,
                                   std::shared_ptr<Object> obj);

template<typename Object, typename Output, typename... Args>
inline void parseSourcesAndOutputs(graph::CallGraph * g,
                                   std::shared_ptr<graph::internal::Inputs> user,
                                   std::shared_ptr<Object> obj,
                                   Output output,
                                   Args &&... args)
{
  user->addInput(obj, output);
  parseSourcesAndOutputs(g, user, obj, std::forward<Args>(args)...);
}

/** Recursion over the sources and outputs.
 * Case where the next element to parse is a source.
 */
template<typename Object1, typename Object2, typename... Args>
inline void parseSourcesAndOutputs(graph::CallGraph * g,
                                   std::shared_ptr<graph::internal::Inputs> user,
                                   std::shared_ptr<Object1>,
                                   std::shared_ptr<Object2> obj2,
                                   Args &&... args)
{
  g->add(user);
  auto newUser = std::make_shared<graph::internal::Inputs>();
  parseSourcesAndOutputs(g, newUser, obj2, std::forward<Args>(args)...);
}

/** Recursion over the sources and outputs.
 * End of recursion.
 */
template<typename Object>
inline void parseSourcesAndOutputs(graph::CallGraph * g,
                                   std::shared_ptr<graph::internal::Inputs> user,
                                   std::shared_ptr<Object>)
{
  g->add(user);
}

} // namespace internal

} // namespace utils

} // namespace tvm
