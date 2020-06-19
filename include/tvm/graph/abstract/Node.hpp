/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#pragma once

#include <tvm/graph/internal/Logger.h>

#include <sstream>


#define TVM_GRAPH_LOG_REGISTER_UPDATE(node, u, fn) \
  tvm::graph::internal::Logger::logger().registerUpdate<U,EnumT>(static_cast<U*>(node), u, fn);

#define TVM_GRAPH_LOG_ADD_OUTPUT_DEPENDENCY(node, o, u) \
  tvm::graph::internal::Logger::logger().addOutputDependency<U,EnumO,EnumU>(static_cast<U*>(node), o, u);

#define TVM_GRAPH_LOG_ADD_INTERNAL_DEPENDENCY(node, uDependent, u) \
  tvm::graph::internal::Logger::logger().addInternalDependency<U,EnumU1,EnumU2>(static_cast<U*>(node), uDependent, u);

#define TVM_GRAPH_LOG_ADD_INPUT_DEPENDENCY(node, u, source, i) \
  tvm::graph::internal::Logger::logger().addInputDependency<U,EnumU,S,EnumO>(static_cast<U*>(node), u, source, i);

#define TVM_GRAPH_LOG_ADD_DIRECT_DEPENDENCY(node, o, source, i) \
  tvm::graph::internal::Logger::logger().addDirectDependency(static_cast<U*>(node), o, source, i);


namespace tvm
{

namespace graph
{

namespace abstract
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
  static_assert(internal::is_valid_update<U>(EnumT()), "This update signal is not part of this Node");
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
  updates_[static_cast<int>(u)] = [fn](AbstractNode & self)
  {
    return (static_cast<U&>(self).*fn)();
  };
  TVM_GRAPH_LOG_REGISTER_UPDATE(this, u, fn)
}

template<typename T>
template<typename U, typename EnumO, typename EnumU>
void Node<T>::addOutputDependency(EnumO o, EnumU u)
{
  static_assert(is_valid_output<U>(EnumO()), "Invalid output for this type. If you are calling this method from a derived class, put this class in template parameter.");
  static_assert(internal::is_valid_update<U>(EnumU()), "Invalid update for this type. If you are calling this method from a derived class, put this class in template parameter.");
  if(!U::OutputStaticallyEnabled(o))
  {
    std::stringstream ss;
    ss << "Output " << U::OutputName(o) << " is disabled within " << U::OutputBaseName;
    throw std::runtime_error(ss.str());
  }
  if (directDependencies_.count(static_cast<int>(o)))
  {
    std::stringstream ss;
    ss << "Output " << U::OutputName(o) << " already has a direct dependency. You cannot mix direct and output dependencies";
    throw std::runtime_error(ss.str());
  }
  outputDependencies_[static_cast<int>(o)].push_back(static_cast<int>(u));
  TVM_GRAPH_LOG_ADD_OUTPUT_DEPENDENCY(this, o, u);
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
  static_assert(internal::is_valid_update<U>(EnumU1()), "Invalid dependent update for this type. If you are calling this method from a derived class, put this class in template parameter.");
  static_assert(internal::is_valid_update<U>(EnumU2()), "Invalid update for this type. If you are calling this method from a derived class, put this class in template parameter.");
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
  TVM_GRAPH_LOG_ADD_INTERNAL_DEPENDENCY(this, uDependent, u)
}

template<typename T>
template<typename U, typename EnumU, typename S>
void Node<T>::checkAddInputDependency(EnumU u)
{
  static_assert(internal::is_valid_update<U>(EnumU()), "Invalid update for this type. If you are calling this method from a derived class, put this class in template parameter.");
  if(!U::UpdateStaticallyEnabled(u))
  {
    std::stringstream ss;
    ss << "Update " << U::UpdateName(u) << " is disabled within " << U::UpdateBaseName;
    throw std::runtime_error(ss.str());
  }
}

template<typename T>
template<typename U, typename EnumU, typename S, typename EnumO, typename ... Args>
void Node<T>::addInputDependency(EnumU u, std::shared_ptr<S> source, EnumO i, Args ... args)
{
  // Make sure the call makes sense
  checkAddInputDependency<U, EnumU, S>(u);
  // Make sure we have this input
  addInput(source, i);
  // Add the dependency
  addInputDependency<U>(u, source.get(), i, args...);
}

template<typename T>
template<typename U, typename EnumU, typename S, typename EnumO, typename ... Args,
    typename std::enable_if<std::is_base_of<abstract::Outputs, S>::value, int>::type>
void Node<T>::addInputDependency(EnumU u, S & source, EnumO i, Args ... args)
{
  // Make sure the call makes sense
  checkAddInputDependency<U, EnumU, S>(u);
  // Make sure we have this input
  addInput(source, i);
  // Add the dependency
  addInputDependency<U>(u, &source, i, args...);
}

template<typename T>
template<typename U, typename EnumU, typename S, typename EnumO>
void Node<T>::addInputDependency(EnumU u, S * source, EnumO i)
{
  static_assert(is_valid_output<S>(EnumO()), "Invalid output for this type of source.");
  if(inputDependencies_.count(static_cast<int>(u)))
  {
    auto & inputDependency = inputDependencies_[static_cast<int>(u)];
    auto depIt = std::find_if(inputDependency.begin(), inputDependency.end(),
                              [&source](const input_dependency_t::value_type & p)
                              {
                                return source == p.first;
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
  TVM_GRAPH_LOG_ADD_INPUT_DEPENDENCY(this, u, source, i);
}

template<typename T>
template<typename U, typename EnumU, typename S, typename EnumO, typename ... Args>
void Node<T>::addInputDependency(EnumU u, S * source, EnumO i, Args ... args)
{
  addInputDependency<U>(u, source, i);
  if(sizeof...(args))
  {
    addInputDependency<U>(u, source, args...);
  }
}

template<typename T>
template<typename U, typename EnumO, typename S, typename EnumI>
void Node<T>::checkAddDirectDependency(EnumO o)
{
  static_assert(is_valid_output<U>(EnumO()), "Invalid output for this type. If you are calling this method from a derived class, put this class in template parameter.");
  static_assert(is_valid_output<S>(EnumI()), "Invalid output for this type of source.");
  if (!U::OutputStaticallyEnabled(o))
  {
    std::stringstream ss;
    ss << "Output " << U::OutputName(o) << " is disabled within " << U::OutputBaseName;
    throw std::runtime_error(ss.str());
  }
  if (directDependencies_.count(static_cast<int>(o)))
  {
    std::stringstream ss;
    ss << "Output " << U::OutputName(o) << " already has a direct dependency. ";
    throw std::runtime_error(ss.str());
  }
  if (outputDependencies_.count(static_cast<int>(o)))
  {
    std::stringstream ss;
    ss << "Output " << U::OutputName(o) << " already has output dependencies. You cannot mix direct and output dependencies";
    throw std::runtime_error(ss.str());
  }
}

template<typename T>
template<typename U, typename EnumO, typename S, typename EnumI>
inline void Node<T>::addDirectDependency(EnumO o, std::shared_ptr<S> source, EnumI i)
{
  // Check the add is valid
  checkAddDirectDependency<U, EnumO, S, EnumI>(o);
  // Make sure we have this input
  addInput(source, i);
  // Add the dependency
  addDirectDependency<U>(o, source.get(), i);
}

template<typename T>
template<typename U, typename EnumO, typename S, typename EnumI,
    typename std::enable_if<std::is_base_of<abstract::Outputs, S>::value, int>::type>
inline void Node<T>::addDirectDependency(EnumO o, S & source, EnumI i)
{
  // Check the add is valid
  checkAddDirectDependency<U, EnumO, S, EnumI>(o);
  // Make sure we have this input
  addInput(source, i);
  // Add the dependency
  addDirectDependency<U>(o, &source, i);
}

template<typename T>
template<typename U, typename EnumO, typename S, typename EnumI>
inline void Node<T>::addDirectDependency(EnumO o, S * source, EnumI i)
{
  directDependencies_[static_cast<int>(o)] = { source, static_cast<int>(i) };
  TVM_GRAPH_LOG_ADD_DIRECT_DEPENDENCY(this, o, source, i);
}

} // namespace abstract

} // namespace graph

} // namespace tvm
