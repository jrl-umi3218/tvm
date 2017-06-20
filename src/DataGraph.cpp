#include "DataGraph.h"

#include <assert.h>
#include <exception>
#include <utility>

namespace taskvm
{

  bool DataUser::hasSource(const DataSource& source) const
  {
    if (sources_.size() > 0)
    {
      // We need to create a shared pointer from the DataSource& to use with find. 
      // Because we don't want source to be delete when the shared_ptr is destroyed,
      // we base its life on an element which will outlive this method.
      //The const_cast is licit: we do not perform non-const operation on source
      std::shared_ptr<DataSource> ptr(*sources_.begin(), const_cast<DataSource*>(&source));

      // could do better (no creation of a temporary shared pointer) in c++14:
      //https://stackoverflow.com/questions/32610446/find-a-value-in-a-set-of-shared-ptr
      //auto ret = sources_.find(std::shared_ptr<DataSource>(const_cast<DataSource*>(&source)));
      auto ret = sources_.find(ptr);
      return ret != sources_.end();
    }
    else 
      return false;
  }

  const std::set<std::shared_ptr<DataSource>>& DataUser::getSources() const
  {
    return sources_;
  }

  const std::vector<internal::UnifiedEnumValue>& DataUser::getInputs(const DataSource& source) const
  {
    static std::vector<internal::UnifiedEnumValue> emptyReturn = {};
    if (hasSource(source))
      return inputs_.at(&source);
    else
      return emptyReturn;
  }

  std::shared_ptr<DataSource> DataUser::getSharedPtr(DataSource const* ds) const
  {
    if (sources_.size() > 0)
    {
      // We need to create a shared pointer from the DataSource& to use with find. 
      // Because we don't want source to be delete when the shared_ptr is destroyed,
      // we base its life on an element which will outlive this method.
      std::shared_ptr<DataSource> ptr(*sources_.begin(), const_cast<DataSource*>(ds));
      // could do better (no creation of a temporary shared pointer) in c++14:
      //https://stackoverflow.com/questions/32610446/find-a-value-in-a-set-of-shared-ptr
      //The const_cast is licit: we do not perform non-const operation on ds
      auto ret = sources_.find(ptr);
      if (ret == sources_.end())
        throw std::range_error("Pointer ds does not refer to a source of this instance.");
      return *ret;
    }
    else
      throw std::range_error("Pointer ds does not refer to a source of this instance.");
  }


  const DataNode::OuputDependencies& DataNode::outputDependencies()
  {
    if (!outputDependenciesAreSet_)
    {
      fillOutputDependencies();
      outputDependenciesAreSet_ = true;
    }
    return outputDependencies_;
  }

  const DataNode::InternalDependencies& DataNode::internalDependencies()
  {
    if (!internalDependenciesAreSet_)
    {
      fillInternalDependencies();
      internalDependenciesAreSet_ = true;
    }
    return internalDependencies_;
  }

  const DataNode::InputDependencies& DataNode::inputDependencies()
  {
    if (!inputDependenciesAreSet_)
    {
      fillUpdateDependencies();
      inputDependenciesAreSet_ = true;
    }
    return inputDependencies_;
  }

  const DataNode::UpdateList& DataNode::outputDependencies(internal::UnifiedEnumValue output)
  {
    static UpdateList emptyReturn;

    if (!outputDependenciesAreSet_)
    {
      fillOutputDependencies();
      outputDependenciesAreSet_ = true;
    }

    auto it = outputDependencies_.find(output);
    if (it == outputDependencies_.end())
      return emptyReturn;
    else
      return it->second;
  }

  const DataNode::UpdateList& DataNode::internalDependencies(internal::UnifiedEnumValue update)
  {
    static UpdateList emptyReturn;

    if (!internalDependenciesAreSet_)
    {
      fillInternalDependencies();
      internalDependenciesAreSet_ = true;
    }

    auto it = internalDependencies_.find(update);
    if (it == internalDependencies_.end())
      return emptyReturn;
    else
      return it->second;
  }

  const DataNode::InputList& DataNode::inputDependencies(internal::UnifiedEnumValue update)
  {
    static InputList emptyReturn;

    if (!inputDependenciesAreSet_)
    {
      fillUpdateDependencies();
      inputDependenciesAreSet_ = true;
    }

    auto it = inputDependencies_.find(update);
    if (it == inputDependencies_.end())
      return emptyReturn;
    else
      return it->second;
  }



  void UpdateGraph::add(std::shared_ptr<DataUser> user)
  {
    for (auto s : user->getSources())
    {
      for (auto i : user->getInputs(*s))
      {
        add(s, i);
      }
    }
  }

  std::vector<int> UpdateGraph::add(std::shared_ptr<DataSource> source, internal::UnifiedEnumValue input)
  {
    auto p = visitedInputs_.find({ source.get(), input });
    if (p != visitedInputs_.end())
      return p->second;
    else
    {
      std::vector<int> updateId;
      auto sourceNode = std::dynamic_pointer_cast<DataNode>(source);
      if (sourceNode)
      {
        auto& outputDependencies = sourceNode->outputDependencies(input);
        for (auto u : outputDependencies)
          updateId.push_back(addUpdate({ sourceNode, u }));
      }
      visitedInputs_[{ source.get(), input }] = updateId;
      return updateId;
    }
  }

  int UpdateGraph::addUpdate(Update u)
  {
    auto it = updateId_.find(u);
    int id;
    if (updateId_.find(u) == updateId_.end())
    {
      id = static_cast<int>(updateId_.size());
      auto p = updateId_.insert({ u, id });
      update_.push_back(const_cast<Update*>(&p.first->first));
      root_.push_back(true);
      dependencies_.push_back({});

      //adding internal dependencies
      auto& internalDependencies = u.nodePtr->internalDependencies(u.updateId);
      for (auto d : internalDependencies)
      {
        int dependencyId = addUpdate({ u.nodePtr, d });
        addEdge(id, dependencyId);
      }

      //adding external dependencies
      auto& inputDependencies = u.nodePtr->inputDependencies(u.updateId);
      for (auto d : inputDependencies)
      {
        std::vector<int> dependenciesId = add(u.nodePtr->getSharedPtr(d.first), d.second);
        for (auto i : dependenciesId)
          addEdge(id, i);
      }

      assert(update_.size() == updateId_.size());
      assert(root_.size() == updateId_.size());
    }
    else
      id = it->second;

    return id;
  }

  void UpdateGraph::addEdge(int from, int to)
  {
    dependencies_[from].push_back(to);
    root_[to] = false; //update #dependencyId has a parent
  }

  UpdatePlan::UpdatePlan(const UpdateGraph& graph)
  {
    buildPlan(graph);
  }

  void UpdatePlan::execute() const
  {
    for (auto& e : plan_)
      e.nodePtr->update(e.updateId);
  }

  void UpdatePlan::buildPlan(const UpdateGraph& graph)
  {
    //number of vertices
    auto n = static_cast<int>(graph.update_.size());

    //initialize temporary data
    order_.clear();
    order_.reserve(n);
    visited_ = std::vector<bool>(n, false);
    stack_ = std::vector<bool>(n, false);

    for (size_t i = 0; i < n; ++i)
    {
      if (graph.root_[i])
        recursiveBuild(graph, i);
    }

    for (auto id : order_)
      plan_.push_back(*graph.update_[id]);
  }

  void UpdatePlan::recursiveBuild(const UpdateGraph& graph, size_t v)
  {
    //code adapted from http://www.geeksforgeeks.org/detect-cycle-in-a-graph/
    if (visited_[v] == false)
    {
      // Mark the current node as visited and part of recursion stack
      visited_[v] = true;
      stack_[v] = true;

      // Recur for all the vertices adjacent to this vertex
      for (auto i : graph.dependencies_[v])
      {
        if (!visited_[i])
          recursiveBuild(graph, i);
        else if (stack_[i])
          throw std::logic_error("The graph contains a cycle");
      }

    }
    stack_[v] = false;  // remove the vertex from recursion stack
    // all the descendants of this this node have been processed, i.e. all the
    // updates this node depends on have been discovered and added to order. No
    // we can push this node.
    order_.push_back(v);  
  }
}
