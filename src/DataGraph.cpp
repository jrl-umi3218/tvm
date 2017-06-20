#include "DataGraph.h"

#include <assert.h>
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
        throw std::exception("Pointer ds does not refer to a source of this instance.");
      return *ret;
    }
    else
      throw std::exception("Pointer ds does not refer to a source of this instance.");
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
      fillOutputDependencies();
      internalDependenciesAreSet_ = true;
    }
    return internalDependencies_;
  }

  const DataNode::InputDependencies& DataNode::inputDependencies()
  {
    if (!inputDependenciesAreSet_)
    {
      fillOutputDependencies();
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
      fillOutputDependencies();
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
      fillOutputDependencies();
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
    auto p = visitedInputs_.insert({ source.get(), input });
    std::vector<int> updateId;
    if (p.second)
    {
      auto sourceNode = std::dynamic_pointer_cast<DataNode>(source);
      if (sourceNode)
      {
        auto& outputDependencies = sourceNode->outputDependencies(input);
        for (auto u : outputDependencies)
          updateId.push_back(addUpdate({ sourceNode, u }));
      }
    }
    return updateId;
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

  void UpdatePlan::execute() const
  {
    for (auto& e : plan_)
      e.first->update(e.second);
  }
}