/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

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