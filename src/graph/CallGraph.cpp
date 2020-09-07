/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/graph/CallGraph.h>
#include <tvm/graph/internal/Logger.h>

#define TVM_GRAPH_LOG_ADD_GRAPH_OUTPUTS(graph, inputs) internal::Logger::logger().addGraphOutput(graph, inputs.get());

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

void CallGraph::update() { plan_.build(*this); }

void CallGraph::clear()
{
  callId_.clear();
  calls_.clear();
  dependencyGraph_.clear();
  visited_.clear();
  plan_.clear();
}

std::vector<int> CallGraph::addOutput(abstract::Outputs * source, int output)
{
  std::intptr_t ptr = reinterpret_cast<std::intptr_t>(source);
  if(visited_.count(ptr) && visited_[ptr].count(output))
  {
    return visited_[ptr][output];
  }
  std::vector<int> callId = {};
  if(source->is_node_)
  {
    auto node = static_cast<internal::AbstractNode *>(source);
    if(node->outputDependencies_.count(output))
    {
      for(const auto & u : node->outputDependencies_[output])
      {
        callId.push_back(addCall({node, u}));
      }
    }
    else if(node->directDependencies_.count(output))
    {
      const auto & p = node->directDependencies_[output];
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
  int id = static_cast<int>(dependencyGraph_.addNode());
  callId_[c] = id;
  calls_.push_back(c);
  if(c.node->internalDependencies_.count(c.id))
  {
    for(auto u : c.node->internalDependencies_[c.id])
    {
      int cId = addCall({c.node, u});
      dependencyGraph_.addEdge(id, cId);
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
          dependencyGraph_.addEdge(id, cId);
        }
      }
    }
  }
  return id;
}

void CallGraph::Plan::build(const CallGraph & graph)
{
  plan_.clear();
  const auto & order = graph.dependencyGraph_.order();

  for(auto i : order)
  {
    plan_.push_back(graph.calls_[i]);
  }
}

void CallGraph::Plan::clear() { plan_.clear(); }

} // namespace graph

} // namespace tvm
