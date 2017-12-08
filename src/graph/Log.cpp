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

namespace tvm
{

namespace graph
{

namespace internal
{
  std::string replaceWhitespaces(std::string name)
  {
    for (int i = 0; i < name.length(); i++)
    {
      if (name[i] == ' ')
        name[i] = '_';
    }
    return name;
  }

  std::string replaceColons(std::string name)
  {
    for (int i = 0; i < name.length(); i++)
    {
      if (name[i] == ':')
        name[i] = '_';
    }
    return name;
  }

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
      abi::__cxa_demangle(name, NULL, NULL, &status),
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

  std::string clean(const std::string& name, bool replaceColon_ = true)
  {
    if (replaceColon_)
    {
      return replaceColons(replaceWhitespaces(demangle(name, true)));
    }
    else
    {
      return replaceWhitespaces(demangle(name, true));
    }
  }

  //Find the index of the output corresponding to a given input
  //Return -1 if it didn't find it
  size_t retrieve(const Log::Input& input, const std::vector<Log::Output>& outputs)
  {
    for (size_t i = 0; i < outputs.size(); ++i)
    {
      const Log::Output& o = outputs[i];
      if (o.owner == input.source && o.id == input.id)
        return i;
    }
    return -1;
  }

  std::string nodeName(const Log::Output& output)
  {
    return "out" + clean(output.name);
  }

  std::string nodeName(const Log::Input& input)
  {
    std::stringstream ss;
    ss << clean(input.source.type.name) << input.source.value << "_in" << clean(input.name);
    return ss.str();
  }

  std::string nodeName(const Log::Update& update)
  {
    return "up" + clean(update.name);
  }

  Log::TypeInfo::TypeInfo(const std::type_info & t)
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

  //find the input corresponding to the input dependency
  const Log::Input& findInput(const std::set<Log::Input>& s, const Log::InputDependency& d)
  {
    for (const auto& i : s)
    {
      if (i.owner.value == d.owner.value && i.id == d.input && i.source == d.source)
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
      if (i.owner.value == d.owner.value && i.id == d.input && i.source == d.source)
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
      if (u.owner.value == d.owner.value && u.id == d.update)
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
      if (u.owner.value == d.owner.value && u.id == d.update)
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
      if (u.owner.value == d.owner.value && u.id == d.from)
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
      if (u.owner.value == d.owner.value && u.id == d.to)
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
      if (o.owner.value == d.owner.value && o.id == d.output)
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
      if (o.owner.value == d.owner.value && o.id == d.output)
      {
        return o;
      }
    }
    throw std::runtime_error("Output not found");
  }


  std::string Log::generateDot(const Pointer& p) const
  {
    std::set<Output> outputs;
    std::set<Update> updates;
    std::map<Pointer, std::set<Input>> inputs;

    //List of outputs
    for (const auto& o : outputs_)
    {
      if (o.owner.value == p.value)
      {
        outputs.insert(o);
      }
    }
    //Ouputs may not be refered by an update, but by the input of another node
    for (const auto& i : inputs_)
    {
      if (i.source.value == p.value)
      {
        outputs.insert({ i.id, i.name, p });
      }
    }

    //List of updates
    for (const auto& u : updates_)
    {
      if (u.owner.value == p.value)
      {
        updates.insert(u);
      }
    }

    //List of inputs
    for (const auto& i : inputs_)
    {
      if (i.owner.value == p.value)
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

} // namespace internal

} // namespace graph

} // namespace tvm