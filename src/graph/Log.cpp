#include <tvm/graph/internal/Log.h>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>

#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
#endif

namespace
{
  using namespace tvm::graph::internal;

  /** Replace the whitespaces in name by underscores*/
  std::string replaceWhitespaces(std::string name)
  {
    for (size_t i = 0; i < name.length(); i++)
    {
      if (name[i] == ' ')
        name[i] = '_';
    }
    return name;
  }

  /** Replace the colons in name by underscores*/
  std::string replaceColons(std::string name)
  {
    for (size_t i = 0; i < name.length(); i++)
    {
      if (name[i] == ':')
        name[i] = '_';
    }
    return name;
  }

  /** Remove from the name the namespace names specified in the static vector below*/
  std::string removeNamespace(const std::string& name)
  {
    static std::vector<std::string> namespaces = { "tvm", "constraint", "function", "graph", "scheme", "task_dynamics", "utils", "abstract", "internal" };
    std::string res = name;
    for (const auto& s : namespaces)
    {
      auto i = res.find(s + "::");
      if (i != std::string::npos)
      {
        res = res.substr(0, i) + res.substr(i + s.size() + 2);
      }
    }
    return res;
  }

  /** Demangle the typeid name*/
  std::string demangle(const std::string& name, bool removeNamespace_ = false)
  {
#if defined(_MSC_VER)
    auto i = name.find("class ");
    if (i != std::string::npos)
    {
      if (removeNamespace_)
      {
        return removeNamespace(name.substr(i + 6));
      }
      else
      {
        return name.substr(i + 6);
      }
    }

    i = name.find("enum ");
    if (i != std::string::npos)
    {
      if (removeNamespace_)
      {
        return removeNamespace(name.substr(i + 5));
      }
      else
      {
        return name.substr(i + 5);
      }
    }

#elif defined(__GNUG__)
    //adapted from https://stackoverflow.com/a/4541470
    int status = -4; // some arbitrary value to eliminate the compiler warning
    std::unique_ptr<char, void(*)(void*)> res{
      abi::__cxa_demangle(name.c_str(), NULL, NULL, &status),
      std::free
    };
    if (status == 0)
    {
      std::string resName = res.get();
      if (removeNamespace_)
      {
        return removeNamespace(name);
      }
    }
#else

#endif
    return name;
  }

  /** Return a clean name from typeid by demagling it, replacing whitespaces and
    * optionaly replacing colons.
    */
  std::string clean(const std::string& name, bool replaceColons_ = true)
  {
    if (replaceColons_)
    {
      return replaceColons(replaceWhitespaces(demangle(name, true)));
    }
    else
    {
      return replaceWhitespaces(demangle(name, true));
    }
  }

  //find the input corresponding to the input dependency
  const Log::Input& findInput(const std::set<Log::Input>& s, const Log::InputDependency& d)
  {
    for (const auto& i : s)
    {
      if (i.owner == d.owner && i.id == d.input && i.source == d.source)
      {
        return i;
      }
    }
    throw std::runtime_error("Input not found");
  }

  //find the input corresponding to the direct dependency
  const Log::Input& findInput(const std::set<Log::Input>& s, const Log::DirectDependency& d)
  {
    for (const auto& i : s)
    {
      if (i.owner == d.owner && i.id == d.input && i.source == d.source)
      {
        return i;
      }
    }
    throw std::runtime_error("Input not found");
  }

  //find the update corresponding to the input dependency
  const Log::Update& findUpdate(const std::set<Log::Update>& s, const Log::InputDependency& d)
  {
    for (const auto& u : s)
    {
      if (u.owner == d.owner && u.id == d.update)
      {
        return u;
      }
    }
    throw std::runtime_error("Update not found");
  }

  //find the update corresponding to the output dependency
  const Log::Update& findUpdate(const std::set<Log::Update>& s, const Log::OutputDependency& d)
  {
    for (const auto& u : s)
    {
      if (u.owner == d.owner && u.id == d.update)
      {
        return u;
      }
    }
    throw std::runtime_error("Update not found");
  }

  //find the update that is the origin of an internal dependency
  const Log::Update& findFromUpdate(const std::set<Log::Update>& s, const Log::InternalDependency& d)
  {
    for (const auto& u : s)
    {
      if (u.owner == d.owner && u.id == d.from)
      {
        return u;
      }
    }
    throw std::runtime_error("Update not found");
  }

  //find the update that is the destination of an internal dependency
  const Log::Update& findToUpdate(const std::set<Log::Update>& s, const Log::InternalDependency& d)
  {
    for (const auto& u : s)
    {
      if (u.owner == d.owner && u.id == d.to)
      {
        return u;
      }
    }
    throw std::runtime_error("Update not found");
  }

  //find the output corresponding to the output dependency
  const Log::Output& findOutput(const std::set<Log::Output>& s, const Log::OutputDependency& d)
  {
    for (const auto& o : s)
    {
      if (o.owner == d.owner && o.id == d.output)
      {
        return o;
      }
    }
    throw std::runtime_error("Output not found");
  }

  //find the output corresponding to the output dependency
  const Log::Output& findOutput(const std::set<Log::Output>& s, const Log::DirectDependency& d)
  {
    for (const auto& o : s)
    {
      if (o.owner == d.owner && o.id == d.output)
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
  Log::TypeInfo::TypeInfo(const std::type_info& t)
    : hash(t.hash_code()), name(t.name())
  {
  }

  std::ostream& operator<<(std::ostream& os, const Log::TypeInfo& t)
  {
    os << "[" << t.hash << " " << t.name << "]";
    return os;
  }

  std::istream& operator>>(std::istream& is, Log::TypeInfo& t)
  {
    char c1, c2;
    is >> c1 >> t.hash >> t.name >> c2;
    if (is.fail())
      throw std::ios_base::failure("Failed to read Logger::TypeInfo");
    return is;
  }

  std::string Log::generateDot(const Pointer& p) const
  {
    std::set<Output> outputs;
    std::set<Update> updates;
    std::map<Pointer, std::set<Input>> inputs;

    //List of outputs
    for (const auto& o : outputs_)
    {
      if (o.owner == p)
      {
        outputs.insert(o);
      }
    }
    //Ouputs may not be refered by an update, but by the input of another node
    for (const auto& i : inputs_)
    {
      if (i.source == p)
      {
        outputs.insert({ i.id, i.name, p });
      }
    }

    //List of updates
    for (const auto& u : updates_)
    {
      if (u.owner == p)
      {
        updates.insert(u);
      }
    }

    //List of inputs
    for (const auto& i : inputs_)
    {
      if (i.owner == p)
      {
        inputs[i.source].insert(i);
      }
    }

    std::stringstream dot;
    dot << "digraph \"" << clean(p.type.name) << "\"\n{\n";
    dot << "  rankdir=\"LR\";\n";

    //inputs
    int c = 0;
    for (const auto& source : inputs)
    {
      dot << "  subgraph cluster" << c++ << " {\n";
      dot << "    label=\"" << clean(source.first.type.name, false) << "\";\n";
      dot << "    node [shape=diamond];\n";
      for (const auto& s : source.second)
      {
        dot << "    " << nodeName(s) << " [label=\"" << clean(s.name) << "\"];\n";
      }
      dot << "  }\n";
    }

    //updates
    dot << "  {\n";
    for (const auto& u : updates)
    {
      dot << "    " << nodeName(u) << " [label=\"" << clean(u.name) << "\"];\n";
    }
    dot << "  }\n";

    //outputs
    dot << "  {\n";
    dot << "    rank=same;\n";
    dot << "    node [shape=octagon];\n";
    for (const auto& o : outputs)
    {
      dot << "    " << nodeName(o) << " [label=\"" << clean(o.name) << "\"];\n";
    }
    dot << "  }\n";

    //input - update links
    for (const auto& d : inputDependencies_)
    {
      if (d.owner.value == p.value)
      {
        auto i = findInput(inputs[d.source], d);
        auto u = findUpdate(updates, d);
        dot << nodeName(i) << "->" << nodeName(u) << ";\n";
      }
    }

    //update - output links
    for (const auto& d : outputDependencies_)
    {
      if (d.owner.value == p.value)
      {
        auto u = findUpdate(updates, d);
        auto o = findOutput(outputs, d);
        dot << nodeName(u) << "->" << nodeName(o) << ";\n";
      }
    }

    //update - update links
    for (const auto& d : internalDependencies_)
    {
      if (d.owner.value == p.value)
      {
        auto f = findFromUpdate(updates, d);
        auto t = findToUpdate(updates, d);
        dot << nodeName(f) << "->" << nodeName(t) << ";\n";
      }
    }

    //input - output links
    for (const auto& d : directDependencies_)
    {
      if (d.owner.value == p.value)
      {
        auto i = findInput(inputs[d.source], d);
        auto o = findOutput(outputs, d);
        dot << nodeName(i) << "->" << nodeName(o) << ";\n";
      }
    }

    dot << "}";
    return dot.str();
  }

  std::string Log::generateDot() const
  {
    std::map<std::uintptr_t, std::set<Output>> outputs;
    std::map<std::uintptr_t, std::set<Update>> updates;
    std::map<std::uintptr_t, std::set<Input>> inputs;
    std::map<Output, bool> isAlsoInput;

    //Sort outputs by owner
    for (const auto& o : outputs_)
    {
      auto p = outputs[o.owner.value].insert(o);
      if (p.second)
      {
        isAlsoInput[o] = false;
      }
    }

    //Sort input by source and process the corresponding output
    for (const auto& i : inputs_)
    {
      inputs[i.source.value].insert(i);
      //Ouputs may not be refered by an update, but by the input of another node
      auto p = outputs[i.source.value].insert({ i.id, i.name, i.source });
      isAlsoInput[*p.first] = true;
    }

    //Sort updates by owner
    for (const auto& u : updates_)
    {
      updates[u.owner.value].insert(u);
    }

    std::stringstream dot;
    dot << "digraph \"" << "Update graph" << "\"\n{\n";
    dot << "  rankdir=\"LR\";\n";

    //Process each node
    int c = 0;
    for (const auto& p : types_)
    {
      dot << " subgraph cluster" << ++c << " {\n";
      dot << "    label=\"" << clean(p.second.back().name, false) << "\";\n";
      //updates
      dot << "    {\n";
      for (const auto& u : updates[p.first])
      {
        dot << "      " << nodeName(u) << " [label=\"" << clean(u.name) << "\"];\n";
      }
      dot << "    }\n";
      //outputs
      dot << "    {\n";
      dot << "      rank=same;\n";
      for (const auto& o : outputs[p.first])
      {
        dot << "      " << nodeName(o) << " [label=\"" << clean(o.name) << "\",";
        if (isAlsoInput[o])
        {
          dot << "shape=Mdiamond";
        }
        else
        {
          dot << "shape=octagon";
        }
        dot << "]; \n";
      }
      dot << "    }\n";
      dot << " }\n";
    }

    //input - update links
    for (const auto& d : inputDependencies_)
    {
      auto i = findInput(inputs[d.source.value], d);
      auto u = findUpdate(updates[d.owner.value], d);
      dot << nodeName(Output{ i.id, i.name, i.source }) << "->" << nodeName(u) << ";\n";
    }

    //update - output links
    for (const auto& d : outputDependencies_)
    {
      auto u = findUpdate(updates[d.owner.value], d);
      auto o = findOutput(outputs[d.owner.value], d);
      dot << nodeName(u) << "->" << nodeName(o) << ";\n";
    }

    //update - update links
    for (const auto& d : internalDependencies_)
    {
      auto f = findFromUpdate(updates[d.owner.value], d);
      auto t = findToUpdate(updates[d.owner.value], d);
      dot << nodeName(f) << "->" << nodeName(t) << ";\n";
    }

    //input - output links
    for (const auto& d : directDependencies_)
    {
      auto i = findInput(inputs[d.source.value], d);
      auto o = findOutput(outputs[d.owner.value], d);
      dot << nodeName(Output{ i.id, i.name, i.source }) << "->" << nodeName(o) << ";\n";
    }

    dot << "}";
    return dot.str();
  }


  std::string Log::nodeName(const Log::Output& output) const
  {
    std::stringstream ss;
    ss << clean(types_.at(output.owner.value).back().name) << output.owner.value << "_out" << clean(output.name);
    return ss.str();
  }

  std::string Log::nodeName(const Log::Input& input) const
  {
    std::stringstream ss;
    ss << clean(types_.at(input.source.value).back().name) << input.source.value << "_in" << clean(input.name);
    return ss.str();
  }

  std::string Log::nodeName(const Log::Update& update) const
  {
    std::stringstream ss;
    ss << clean(types_.at(update.owner.value).back().name) << update.owner.value << "_up" << clean(update.name);
    return ss.str();
  }

} // namespace internal

} // namespace graph

} // namespace tvm