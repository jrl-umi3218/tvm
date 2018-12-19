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

#include <tvm/graph/CallGraph.h>
#include <tvm/graph/internal/Logger.h>

#define TVM_GRAPH_LOG_ADD_GRAPH_OUTPUTS(graph, inputs) \
  internal::Logger::logger().addGraphOutput(graph, inputs.get());

namespace tvm
{

namespace graph
{

void CallGraph::add(std::shared_ptr<internal::Inputs> inputs)
{
  inputs_.push_back(inputs);
  for(auto & i : inputs->inputs_)
  {
    auto & source = i.first;
    auto & outputs = i.second;
    for(auto o : outputs)
    {
      addOutput(source, o);
    }
  }
  TVM_GRAPH_LOG_ADD_GRAPH_OUTPUTS(this, inputs)
}

void CallGraph::update()
{
  plan_.build(*this);
}

void CallGraph::clear()
{
  callId_.clear();
  calls_.clear();
  dependencies_.clear();
  root_.clear();
  visited_.clear();
  plan_.clear();
}

std::vector<int> CallGraph::addOutput(abstract::Outputs * source,
                                      int output)
{
  std::intptr_t ptr = reinterpret_cast<std::intptr_t>(source);
  if(visited_.count(ptr) && visited_[ptr].count(output))
  {
    return visited_[ptr][output];
  }
  std::vector<int> callId = {};
  if(source->is_node_)
  {
    auto node = static_cast<internal::AbstractNode*>(source);
    if(node->outputDependencies_.count(output))
    {
      for(const auto & u : node->outputDependencies_[output])
      {
        callId.push_back(addCall({node, u}));
      }
    }
    else if (node->directDependencies_.count(output))
    {
      const auto& p = node->directDependencies_[output];
      callId = addOutput(p.first, p.second);
    }
  }
  if(!visited_.count(ptr))
  {
    visited_[ptr] = {};
  }
  visited_[ptr][output] = callId;
  return callId;
}

int CallGraph::addCall(Call c)
{
  if(callId_.count(c))
  {
    return callId_[c];
  }
  int id = static_cast<int>(callId_.size());
  callId_[c] = id;
  calls_.push_back(c);
  root_.push_back(true);
  dependencies_.push_back({});
  if(c.node->internalDependencies_.count(c.id))
  {
    for(auto u : c.node->internalDependencies_[c.id])
    {
      int cId = addCall({c.node, u});
      addEdge(id, cId);
    }
  }
  if(c.node->inputDependencies_.count(c.id))
  {
    for(auto i : c.node->inputDependencies_[c.id])
    {
      for(auto o : i.second)
      {
        std::vector<int> cIds = addOutput(i.first, o);
        for(auto cId : cIds)
        {
          addEdge(id, cId);
        }
      }
    }
  }
  return id;
}

void CallGraph::addEdge(int from, int to)
{
  assert(static_cast<size_t>(from) < dependencies_.size());
  assert(static_cast<size_t>(to) < root_.size());
  dependencies_[from].push_back(to);
  root_[to] = false;
}

void CallGraph::Plan::build(const CallGraph & graph)
{
  plan_.clear();
  std::vector<size_t> order;
  order.reserve(graph.calls_.size());
  std::vector<bool> visited(order.capacity(), false);
  std::vector<bool> stack(order.capacity(), false);

  bool has_root = false;
  for(size_t i = 0; i < visited.size(); ++i)
  {
    if(graph.root_[i])
    {
      recursiveBuild(graph, i, order, visited, stack);
      has_root = true;
    }
  }

  if(!has_root && visited.size() != 0)
  {
    throw std::logic_error("Try to build a plan on a non-empty graph with no root. It contains at least one cycle.");
  }

  for(auto i : order)
  {
    plan_.push_back(graph.calls_[i]);
  }
}

void CallGraph::Plan::clear()
{
  plan_.clear();
}

void CallGraph::Plan::recursiveBuild(const CallGraph & graph, size_t v,
                                     std::vector<size_t> & order,
                                     std::vector<bool> & visited,
                                     std::vector<bool> & stack)
{
  if(!visited[v])
  {
    visited[v] = true;
    stack[v] = true;

    for(auto i : graph.dependencies_[v])
    {
      if(!visited[i]) { recursiveBuild(graph, i, order, visited, stack); }
      else if(stack[i])
      {
        throw std::logic_error("The graph contains a cycle");
      }
    }
  }
  stack[v] = false;
  order.push_back(v);
}

} // namespace graph

} // namespace tvm
