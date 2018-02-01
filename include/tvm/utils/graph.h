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
#include <tvm/utils/internal/graphDetails.h>
#include <tvm/graph/internal/Inputs.h>

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
  inline std::unique_ptr<graph::CallGraph> generateUpdateGraph(std::shared_ptr<Object> obj, Args&&... args);

  template<typename Object, typename... Args>
  inline std::unique_ptr<graph::CallGraph> generateUpdateGraph(std::shared_ptr<Object> obj, Args&&... args)
  {
    auto ptr = std::unique_ptr<graph::CallGraph>(new graph::CallGraph());
    auto user = std::make_shared<graph::internal::Inputs>();
    internal::parseSourcesAndOutputs(ptr.get(), user, obj, std::forward<Args>(args)...);
    ptr->update();

    return ptr;
  }

}

}