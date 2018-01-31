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
  /** Generate the graph of updates for obj, given a list of outputs*/
  template<typename Object, typename... Outputs>
  inline std::unique_ptr<graph::CallGraph> generateUpdateGraph(std::shared_ptr<Object> obj, Outputs&&... outputs);

  template<typename Object, typename... Outputs>
  inline std::unique_ptr<graph::CallGraph> generateUpdateGraph(std::shared_ptr<Object> obj, Outputs&&... outputs)
  {
    auto user = std::make_shared<graph::internal::Inputs>();
    user->addInput(obj, std::forward<Outputs>(outputs)...);
    auto ptr = std::unique_ptr<graph::CallGraph>(new graph::CallGraph());
    ptr->add(user);
    ptr->update();

    return ptr;
  }

}

}

}