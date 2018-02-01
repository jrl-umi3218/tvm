#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

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
  inline void parseSourcesAndOutputs(graph::CallGraph* g, std::shared_ptr<graph::internal::Inputs> user, 
                                     std::shared_ptr<Object> obj, Output output, Args&&... args);

  /** Recursion over the sources and outputs. 
    * Case where the next element to parse is a source.
    */
  template<typename Object1, typename Object2, typename... Args>
  inline void parseSourcesAndOutputs(graph::CallGraph* g, std::shared_ptr<graph::internal::Inputs> user,
                                     std::shared_ptr<Object1> obj1, std::shared_ptr<Object2> obj2, Args&&... args);
  
  /** Recursion over the sources and outputs. 
    * End of recursion.
    */
  template<typename Object>
  inline void parseSourcesAndOutputs(graph::CallGraph* g, std::shared_ptr<graph::internal::Inputs> user, std::shared_ptr<Object> obj);
  
  template<typename Object, typename Output, typename... Args>
  inline void parseSourcesAndOutputs(graph::CallGraph* g, std::shared_ptr<graph::internal::Inputs> user,
    std::shared_ptr<Object> obj, Output output, Args&&... args)
  {
    user->addInput(obj, output);
    parseSourcesAndOutputs(g, user, obj, std::forward<Args>(args)...);
  }

  /** Recursion over the sources and outputs.
  * Case where the next element to parse is a source.
  */
  template<typename Object1, typename Object2, typename... Args>
  inline void parseSourcesAndOutputs(graph::CallGraph* g, std::shared_ptr<graph::internal::Inputs> user,
    std::shared_ptr<Object1> obj1, std::shared_ptr<Object2> obj2, Args&&... args)
  {
    g->add(user);
    auto newUser = std::make_shared<graph::internal::Inputs>();
    parseSourcesAndOutputs(g, newUser, obj2, std::forward<Args>(args)...);
  }

  /** Recursion over the sources and outputs.
  * End of recursion.
  */
  template<typename Object>
  inline void parseSourcesAndOutputs(graph::CallGraph* g, std::shared_ptr<graph::internal::Inputs> user, std::shared_ptr<Object> obj)
  {
    g->add(user);
  }

}

}

}