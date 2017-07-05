#pragma once

#include <sstream>

namespace tvm
{

namespace data
{

template<typename T>
template<typename EnumT, typename U, typename ... Args>
void Node<T>::registerUpdates(EnumT u, void(U::*fn)(), Args ... args)
{
  registerUpdates(u, fn);
  if(sizeof...(args))
  {
    registerUpdates(args...);
  }
}

template<typename T>
template<typename EnumT, typename U>
void Node<T>::registerUpdates(EnumT u, void(U::*fn)())
{
  static_assert(is_valid_update<U>(EnumT()), "This update signal is not part of this Node");
  if(!U::UpdateStaticallyEnabled(u))
  {
    std::stringstream ss;
    ss << "Update " << U::UpdateName(u) << " is disabled within " << U::UpdateBaseName;
    throw std::runtime_error(ss.str());
  }
  // FIXME Assert? Safe-mode? Allow to overwrite function calls this way?
  if(updates_.count(static_cast<int>(u)))
  {
    throw std::range_error("Attempted to register an update call using an id that was already registered.");
  }
  updates_[static_cast<int>(u)] = [this,fn]()
  {
    return (static_cast<U*>(this)->*fn)();
  };
}

template<typename T>
template<typename U, typename EnumO, typename EnumU>
void Node<T>::addOutputDependency(EnumO o, EnumU u)
{
  static_assert(is_valid_output<U>(EnumO()), "Invalid output for this type. If you are calling this method from a derived class, put this class in template parameter.");
  static_assert(is_valid_update<U>(EnumU()), "Invalid update for this type. If you are calling this method from a derived class, put this class in template parameter.");
  if(!U::OutputStaticallyEnabled(o))
  {
    std::stringstream ss;
    ss << "Output " << U::OutputName(o) << " is disabled within " << U::OutputBaseName;
    throw std::runtime_error(ss.str());
  }
  outputDependencies_[static_cast<int>(o)].push_back(static_cast<int>(u));
}

template<typename T>
template<typename U, typename EnumO, typename EnumU>
void Node<T>::addOutputDependency(std::initializer_list<EnumO> os, EnumU u)
{
  for(auto o : os)
  {
    addOutputDependency<U>(o, u);
  }
}

template<typename T>
template<typename U, typename EnumU1, typename EnumU2>
void Node<T>::addInternalDependency(EnumU1 uDependent, EnumU2 u)
{
  static_assert(is_valid_update<U>(EnumU1()), "Invalid dependent update for this type.");
  static_assert(is_valid_update<U>(EnumU2()), "Invalid update for this type.");
  if(!U::UpdateStaticallyEnabled(uDependent))
  {
    std::stringstream ss;
    ss << "Update " << U::UpdateName(uDependent) << " is disabled within " << U::UpdateBaseName;
    throw std::runtime_error(ss.str());
  }
  if(!U::UpdateStaticallyEnabled(u))
  {
    std::stringstream ss;
    ss << "Update " << U::UpdateName(u) << " is disabled within " << U::UpdateBaseName;
    throw std::runtime_error(ss.str());
  }
  internalDependencies_[static_cast<int>(uDependent)].push_back(static_cast<int>(u));
}

template<typename T>
template<typename U, typename EnumU, typename S, typename EnumO>
void Node<T>::addInputDependency(EnumU u, std::shared_ptr<S> source, EnumO i)
{
  static_assert(is_valid_update<U>(EnumU()), "Invalid update for this type.");
  static_assert(is_valid_output<S>(EnumO()), "Invalid output for this type of source.");
  if(!U::UpdateStaticallyEnabled(u))
  {
    std::stringstream ss;
    ss << "Update " << U::UpdateName(u) << " is disabled within " << U::UpdateBaseName;
    throw std::runtime_error(ss.str());
  }
  // Make sure we have this input
  addInput(source, i);
  if(inputDependencies_.count(static_cast<int>(u)))
  {
    auto & inputDependency = inputDependencies_[static_cast<int>(u)];
    auto depIt = std::find_if(inputDependency.begin(), inputDependency.end(),
                              [&source](const input_dependency_t::value_type & p)
                              {
                                return source.get() == p.first.get();
                              });
    if(depIt == inputDependency.end())
    {
      inputDependency[source] = { static_cast<int>(i) };
    }
    else
    {
      depIt->second.insert(static_cast<int>(i));
    }
  }
  else
  {
    inputDependencies_[static_cast<int>(u)] = {{source, {static_cast<int>(i)}}};
  }
}

template<typename T>
template<typename U, typename EnumU, typename S, typename EnumO, typename ... Args>
void Node<T>::addInputDependency(EnumU u, std::shared_ptr<S> source, EnumO i, Args ... args)
{
  addInputDependency<U>(u, source, i);
  if(sizeof...(args))
  {
    addInputDependency<U>(u, source, args...);
  }
}

} // namespace data

} // namespace tvm
