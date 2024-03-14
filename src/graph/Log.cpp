/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/graph/CallGraph.h>
#include <tvm/graph/internal/Log.h>

#include <algorithm>
#include <map>
#include <sstream>

#ifdef __GNUG__
#  include <cstdlib>
#  include <cxxabi.h>
#  include <memory>
#endif

namespace
{
using namespace tvm::graph::internal;

/** Replace the spaces in name by underscores*/
std::string replaceSpaces(std::string name)
{
  for(size_t i = 0; i < name.length(); i++)
  {
    if(name[i] == ' ')
      name[i] = '_';
  }
  return name;
}

/** Replace the colons in name by underscores*/
std::string replaceColons(std::string name)
{
  for(size_t i = 0; i < name.length(); i++)
  {
    if(name[i] == ':')
      name[i] = '_';
  }
  return name;
}

/** Remove from the name the namespace names specified in the static vector below*/
std::string removeNamespace(const std::string & name)
{
  static std::vector<std::string> namespaces = {"tvm",           "constraint", "function", "graph",   "scheme",
                                                "task_dynamics", "utils",      "abstract", "internal"};
  std::string res = name;
  for(const auto & s : namespaces)
  {
    auto i = res.find(s + "::");
    if(i != std::string::npos)
    {
      res = res.substr(0, i) + res.substr(i + s.size() + 2);
    }
  }
  return res;
}

/** De-mangle the typeid name*/
std::string demangle(const std::string & name, bool removeNamespace_ = false)
{
#if defined(_MSC_VER)
  auto i = name.find("class ");
  if(i != std::string::npos)
  {
    if(removeNamespace_)
    {
      return removeNamespace(name.substr(i + 6));
    }
    else
    {
      return name.substr(i + 6);
    }
  }

  i = name.find("enum ");
  if(i != std::string::npos)
  {
    if(removeNamespace_)
    {
      return removeNamespace(name.substr(i + 5));
    }
    else
    {
      return name.substr(i + 5);
    }
  }

#elif defined(__GNUG__)
  // adapted from https://stackoverflow.com/a/4541470
  int status = -4; // some arbitrary value to eliminate the compiler warning
  std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free};
  if(status == 0)
  {
    std::string resName = res.get();
    if(removeNamespace_)
    {
      return removeNamespace(resName);
    }
    else
    {
      return resName;
    }
  }
  else
  {
    return name;
  }
#else

#endif
  return name;
}

/** Return a clean name from typeid by demangling it, replacing spaces and
 * optional replacing colons.
 */
std::string clean(const std::string & name, bool replaceColons_ = true)
{
  if(replaceColons_)
  {
    return replaceColons(replaceSpaces(demangle(name, true)));
  }
  else
  {
    return replaceSpaces(demangle(name, true));
  }
}

// find the input corresponding to the input dependency
template<typename InputContainer>
const Log::Input & findInput(const InputContainer & s, const Log::InputDependency & d)
{
  for(const auto & i : s)
  {
    if(i.owner == d.owner && i.id == d.input && i.source == d.source)
    {
      return i;
    }
  }
  throw std::runtime_error("Input not found");
}

// find the input corresponding to the direct dependency
template<typename InputContainer>
const Log::Input & findInput(const InputContainer & s, const Log::DirectDependency & d)
{
  for(const auto & i : s)
  {
    if(i.owner == d.owner && i.id == d.input && i.source == d.source)
    {
      return i;
    }
  }
  throw std::runtime_error("Input not found");
}

// find the update corresponding to the input dependency
const Log::Update & findUpdate(const std::set<Log::Update> & s, const Log::InputDependency & d)
{
  for(const auto & u : s)
  {
    if(u.owner == d.owner && u.id == d.update)
    {
      return u;
    }
  }
  throw std::runtime_error("Update not found");
}

// find the update corresponding to the output dependency
template<typename UpdateContainer>
const Log::Update & findUpdate(const UpdateContainer & s, const Log::OutputDependency & d)
{
  for(const auto & u : s)
  {
    if(u.owner == d.owner && u.id == d.update)
    {
      return u;
    }
  }
  throw std::runtime_error("Update not found");
}

// find the update that is the origin of an internal dependency
template<typename UpdateContainer>
const Log::Update & findFromUpdate(const UpdateContainer & s, const Log::InternalDependency & d)
{
  for(const auto & u : s)
  {
    if(u.owner == d.owner && u.id == d.from)
    {
      return u;
    }
  }
  throw std::runtime_error("Update not found");
}

// find the update that is the destination of an internal dependency
const Log::Update & findToUpdate(const std::set<Log::Update> & s, const Log::InternalDependency & d)
{
  for(const auto & u : s)
  {
    if(u.owner == d.owner && u.id == d.to)
    {
      return u;
    }
  }
  throw std::runtime_error("Update not found");
}

// find the output corresponding to the output dependency
const Log::Output & findOutput(const std::set<Log::Output> & s, const Log::OutputDependency & d)
{
  for(const auto & o : s)
  {
    if(o.owner == d.owner && o.id == d.output)
    {
      return o;
    }
  }
  throw std::runtime_error("Output not found");
}

// find the output corresponding to the output dependency
const Log::Output & findOutput(const std::set<Log::Output> & s, const Log::DirectDependency & d)
{
  for(const auto & o : s)
  {
    if(o.owner == d.owner && o.id == d.output)
    {
      return o;
    }
  }
  throw std::runtime_error("Output not found");
}

// find the output corresponding to an input
const Log::Output & findOutput(const std::vector<Log::Output> & v, const Log::Input & i)
{
  for(const auto & o : v)
  {
    if(o.owner == i.source && o.id == i.id)
    {
      return o;
    }
  }
  throw std::runtime_error("Output not found");
}

} // anonymous namespace

namespace tvm
{

namespace graph
{

namespace internal
{
std::ostream & operator<<(std::ostream & os, const std::type_index & t)
{
  os << "[" << t.hash_code() << " " << t.name() << "]";
  return os;
}

std::pair<std::vector<Log::Output>, std::vector<Log::Update>> Log::subGraph(const CallGraph * const g) const
{
  auto it = graphOutputs_.find(g);
  if(it == graphOutputs_.end())
  {
    return {{}, {}};
  }

  std::vector<Output> outputs = outputs_;
  // Adding outputs referred only as source
  for(const auto & i : inputs_)
  {
    outputs.push_back({i.id, i.name, i.source});
  }

  // first we retrieve the outputs of the call graph
  std::vector<Output> startingPoints;
  for(const auto & p : it->second)
  {
    for(const auto & i : inputs_)
    {
      if(i.owner == p)
      {
        startingPoints.push_back(findOutput(outputs, i));
      }
    }
  }
  return followUpDependency(outputs, startingPoints);
}

std::pair<std::vector<Log::Output>, std::vector<Log::Update>> Log::subGraph(const Output out) const
{
  std::vector<Output> outputs = outputs_;
  // Adding outputs referred only as source
  for(const auto & i : inputs_)
  {
    outputs.push_back({i.id, i.name, i.source});
  }

  std::vector<Output> startingPoints = {out};
  return followUpDependency(outputs, startingPoints);
}

std::string Log::generateDot(const Pointer & p) const
{
  std::set<Output> outputs;
  std::set<Update> updates;
  std::map<Pointer, std::set<Input>> inputs;

  // List of outputs
  for(const auto & o : outputs_)
  {
    if(o.owner == p)
    {
      outputs.insert(o);
    }
  }
  // Outputs may not be referred by an update, but by the input of another node
  for(const auto & i : inputs_)
  {
    if(i.source == p)
    {
      outputs.insert({i.id, i.name, p});
    }
  }

  // List of updates
  for(const auto & u : updates_)
  {
    if(u.owner == p)
    {
      updates.insert(u);
    }
  }

  // List of inputs
  for(const auto & i : inputs_)
  {
    if(i.owner == p)
    {
      inputs[i.source].insert(i);
    }
  }

  std::stringstream dot;
  dot << "digraph \"" << clean(p.type.name()) << "\"\n{\n";
  dot << "  rankdir=\"LR\";\n";

  // inputs
  int c = 0;
  for(const auto & source : inputs)
  {
    dot << "  subgraph cluster" << c++ << " {\n";
    dot << "    label=\"" << clean(source.first.type.name(), false) << "\";\n";
    dot << "    node [shape=diamond];\n";
    for(const auto & s : source.second)
    {
      dot << "    " << nodeName(s) << " [label=\"" << clean(s.name) << "\"];\n";
    }
    dot << "  }\n";
  }

  // updates
  dot << "  {\n";
  for(const auto & u : updates)
  {
    dot << "    " << nodeName(u) << " [label=\"" << clean(u.name) << "\"];\n";
  }
  dot << "  }\n";

  // outputs
  dot << "  {\n";
  dot << "    rank=same;\n";
  dot << "    node [shape=octagon];\n";
  for(const auto & o : outputs)
  {
    dot << "    " << nodeName(o) << " [label=\"" << clean(o.name) << "\"];\n";
  }
  dot << "  }\n";

  // input - update links
  for(const auto & d : inputDependencies_)
  {
    if(d.owner.value == p.value)
    {
      auto i = findInput(inputs[d.source], d);
      auto u = findUpdate(updates, d);
      dot << nodeName(i) << "->" << nodeName(u) << ";\n";
    }
  }

  // update - output links
  for(const auto & d : outputDependencies_)
  {
    if(d.owner.value == p.value)
    {
      auto u = findUpdate(updates, d);
      auto o = findOutput(outputs, d);
      dot << nodeName(u) << "->" << nodeName(o) << ";\n";
    }
  }

  // update - update links
  for(const auto & d : internalDependencies_)
  {
    if(d.owner.value == p.value)
    {
      auto f = findFromUpdate(updates, d);
      auto t = findToUpdate(updates, d);
      dot << nodeName(f) << "->" << nodeName(t) << ";\n";
    }
  }

  // input - output links
  for(const auto & d : directDependencies_)
  {
    if(d.owner.value == p.value)
    {
      auto i = findInput(inputs[d.source], d);
      auto o = findOutput(outputs, d);
      dot << nodeName(i) << "->" << nodeName(o) << ";\n";
    }
  }

  dot << "}";
  return dot.str();
}

std::string Log::generateDot(const CallGraph * const g) const
{
  const auto & [outputsInGraph, updatesInGraph] = subGraph(g);
  if(!outputsInGraph.empty() || !updatesInGraph.empty())
  {
    return generateDot(outputsInGraph, updatesInGraph);
  }
  else
  {
    return "";
  }
}

std::string Log::generateDot(const std::vector<Log::Output> & outHighlight,
                             const std::vector<Log::Update> & upHighlight) const
{
  std::map<std::uintptr_t, std::set<Output>> outputs;
  std::map<std::uintptr_t, std::set<Update>> updates;
  std::map<std::uintptr_t, std::set<Input>> inputs;
  std::map<Output, bool> isAlsoInput;
  std::map<Output, bool> oh;
  std::map<Update, bool> uh;

  // Sort outputs by owner
  for(const auto & o : outputs_)
  {
    auto p = outputs[o.owner.value].insert(o);
    if(p.second)
    {
      isAlsoInput[o] = false;
    }
  }

  // Sort input by source and process the corresponding output
  for(const auto & i : inputs_)
  {
    inputs[i.source.value].insert(i);
    // Outputs may not be referred by an update, but by the input of another node
    auto p = outputs[i.source.value].insert({i.id, i.name, i.source});
    isAlsoInput[*p.first] = true;
  }

  // Sort updates by owner
  for(const auto & u : updates_)
  {
    updates[u.owner.value].insert(u);
  }

  // process the highlights
  for(auto & o : outHighlight)
  {
    oh[o] = true;
  }
  for(auto & u : upHighlight)
  {
    uh[u] = true;
  }

  // generate the dot code
  std::stringstream dot;
  dot << "digraph \"" << "Update graph" << "\"\n{\n";
  dot << "  rankdir=\"LR\";\n";

  // Process each node
  int c = 0;
  for(const auto & p : types_)
  {
    dot << " subgraph cluster" << ++c << " {\n";
    dot << "    label=\"" << clean(p.second.back().name(), false) << "\";\n";
    // updates
    dot << "    {\n";
    dot << "      rank=same;\n";
    for(const auto & u : updates[p.first])
    {
      dot << "      " << nodeName(u) << " [label=\"" << clean(u.name) << "\"";
      if(uh[u])
      {
        dot << ",color=orange";
      }
      dot << "];\n";
    }
    dot << "    }\n";
    // outputs
    dot << "    {\n";
    dot << "      rank=same;\n";
    for(const auto & o : outputs[p.first])
    {
      dot << "      " << nodeName(o) << " [label=\"" << clean(o.name) << "\",";
      if(isAlsoInput[o])
      {
        dot << "shape=Mdiamond";
      }
      else
      {
        dot << "shape=octagon";
      }
      if(oh[o])
      {
        dot << ",color=orange";
      }
      dot << "]; \n";
    }
    dot << "    }\n";
    dot << " }\n";
  }

  // input - update links
  for(const auto & d : inputDependencies_)
  {
    auto i = findInput(inputs[d.source.value], d);
    auto u = findUpdate(updates[d.owner.value], d);
    Output o{i.id, i.name, i.source};
    dot << nodeName(o) << "->" << nodeName(u);
    if(oh[o] && uh[u])
    {
      dot << " " << "[color=orange]";
    }
    dot << ";\n";
  }

  // update - output links
  for(const auto & d : outputDependencies_)
  {
    auto u = findUpdate(updates[d.owner.value], d);
    auto o = findOutput(outputs[d.owner.value], d);
    dot << nodeName(u) << "->" << nodeName(o);
    if(oh[o] && uh[u])
    {
      dot << " " << "[color=orange]";
    }
    dot << ";\n";
  }

  // update - update links
  for(const auto & d : internalDependencies_)
  {
    auto from = findFromUpdate(updates[d.owner.value], d);
    auto to = findToUpdate(updates[d.owner.value], d);
    dot << nodeName(from) << "->" << nodeName(to);
    if(uh[from] && uh[to])
    {
      dot << " " << "[color=orange]";
    }
    dot << ";\n";
  }

  // input - output links
  for(const auto & d : directDependencies_)
  {
    auto i = findInput(inputs[d.source.value], d);
    auto to = findOutput(outputs[d.owner.value], d);
    Output from{i.id, i.name, i.source};
    dot << nodeName(from) << "->" << nodeName(to);
    if(oh[from] && oh[to])
    {
      dot << " " << "[color=orange]";
    }
    dot << ";\n";
  }

  dot << "}";
  return dot.str();
}

const std::type_index & Log::getPromotedType(const Pointer & p) const
{
  auto it = types_.find(p.value);
  if(it != types_.end())
    return it->second.back();
  else
    return p.type;
}

std::pair<std::vector<Log::Output>, std::vector<Log::Update>> Log::followUpDependency(
    const std::vector<Output> & allOutputs,
    const std::vector<Output> & startingPoints) const
{
  // We want to populate the following vectors with the updates and outputs linked to startingPoints
  std::vector<Output> outputsInGraph;
  std::vector<Update> updatesInGraph;

  std::map<Output, bool> processedOutputs;
  std::map<Update, bool> processedUpdates;
  std::vector<Output> outputStack = startingPoints;
  std::vector<Update> updateStack;

  // we follow the dependency backward
  while(!outputStack.empty() || !updateStack.empty())
  {
    if(!outputStack.empty())
    {
      auto o = outputStack.back();
      outputStack.pop_back();
      if(!processedOutputs[o])
      {
        outputsInGraph.push_back(o);
        for(const auto & d : outputDependencies_)
        {
          if(d.owner == o.owner && d.output == o.id)
          {
            updateStack.push_back(findUpdate(updates_, d));
          }
        }
        for(const auto & d : directDependencies_)
        {
          if(d.owner == o.owner && d.output == o.id)
          {
            auto i = findInput(inputs_, d);
            outputStack.push_back(findOutput(allOutputs, i));
          }
        }
        processedOutputs[o] = true;
      }
    }

    if(!updateStack.empty())
    {
      auto u = updateStack.back();
      updateStack.pop_back();
      if(!processedUpdates[u])
      {
        updatesInGraph.push_back(u);
        for(const auto & d : inputDependencies_)
        {
          if(d.owner == u.owner && d.update == u.id)
          {
            auto i = findInput(inputs_, d);
            outputStack.push_back(findOutput(allOutputs, i));
          }
        }
        for(const auto & d : internalDependencies_)
        {
          if(d.owner == u.owner && d.to == u.id)
          {
            updateStack.push_back(findFromUpdate(updates_, d));
          }
        }
        processedUpdates[u] = true;
      }
    }
  }
  return {outputsInGraph, updatesInGraph};
}

std::string Log::nodeName(const Log::Output & output) const
{
  std::stringstream ss;
  ss << clean(types_.at(output.owner.value).back().name()) << output.owner.value << "_out" << clean(output.name);
  return ss.str();
}

std::string Log::nodeName(const Log::Input & input) const
{
  std::stringstream ss;
  ss << clean(types_.at(input.source.value).back().name()) << input.source.value << "_in" << clean(input.name);
  return ss.str();
}

std::string Log::nodeName(const Log::Update & update) const
{
  std::stringstream ss;
  ss << clean(types_.at(update.owner.value).back().name()) << update.owner.value << "_up" << clean(update.name);
  return ss.str();
}

} // namespace internal

} // namespace graph

} // namespace tvm
