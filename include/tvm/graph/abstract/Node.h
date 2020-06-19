/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#pragma once

#include <tvm/graph/internal/AbstractNode.h>

namespace tvm
{

namespace graph
{

namespace abstract
{

/** Node is the concrete base for AbstractNode.
 *
 * It represents an entity that uses input signals to compute output
 * signals using update functions and the dependencies between these
 * quantitites.
 *
 */
template<typename T>
class Node : public internal::AbstractNode
{
protected:
  /** Register updates
   *
   * An update is formed by its unique id (from an enum type) and the
   * related object method.
   *
   */
  template<typename EnumT, typename U, typename ... Args>
  void registerUpdates(EnumT u, void(U::*fn)(), Args ... args);

  /** Register a single update */
  template<typename EnumT, typename U>
  void registerUpdates(EnumT u, void(U::*fn)());

  /** Add a dependency of an output to an update call
   *
   * It is not possible for an output to have a direct and some update
   * dependencies at the same time.
   *
   */
  template<typename U = T, typename EnumO, typename EnumU>
  void addOutputDependency(EnumO o, EnumU u);

  /** Add a dependency of multiple outputs to an update call */
  template<typename U = T, typename EnumO, typename EnumU>
  void addOutputDependency(std::initializer_list<EnumO> os, EnumU u);

  /** Add a dependency between two update calls.
   *
   * The first argument of this function depends on the second argument.
   *
   */
  template<typename U = T, typename EnumU1, typename EnumU2>
  void addInternalDependency(EnumU1 uDependent, EnumU2 u);

  /** Add a dependency of an update function to multiple input signals from a source */
  template<typename U = T, typename EnumU, typename S, typename EnumO, typename ... Args>
  void addInputDependency(EnumU u, std::shared_ptr<S> source, EnumO i, Args ... args);

  /** Add a dependency of an update function to multiple input signals from a source
   *
   * The lifetime of \p source should be guaranteed by the caller
   *
   */
  template<typename U = T, typename EnumU, typename S, typename EnumO, typename ... Args,
    typename std::enable_if<std::is_base_of<abstract::Outputs, S>::value, int>::type = 0 >
  void addInputDependency(EnumU u, S & source, EnumO i, Args ... args);

  /** Add a dependency of an output to an input signal and its source
   *
   * This is used when an output directly use the input signal without
   * requiring an update, and a cache. This is the case when an output directly
   * forward an input, and it should be the only use-case.
   *
   * It is not possible for an output to have a direct and some update
   * dependencies at the same time.
   *
   */
  template<typename U = T, typename EnumO, typename S, typename EnumI>
  void addDirectDependency(EnumO o, std::shared_ptr<S> source, EnumI i);

  /** Add a dependency of an output to an input signal and its source
   *
   * The lifetime of \p source should be guaranteed by the caller
   *
   * This is used when an output directly use the input signal without
   * requiring an update, and a cache. This is the case when an output directly
   * forward an input, and it should be the only use-case.
   *
   * It is not possible for an output to have a direct and some update
   * dependencies at the same time.
   *
   */
  template<typename U = T, typename EnumO, typename S, typename EnumI,
    typename std::enable_if<std::is_base_of<abstract::Outputs, S>::value, int>::type = 0 >
  void addDirectDependency(EnumO o, S & source, EnumI i);
private:
  /* Internal version that takes a pointer to the source */
  template<typename U, typename EnumU, typename S>
  void checkAddInputDependency(EnumU u);
  template<typename U = T, typename EnumU, typename S, typename EnumO>
  void addInputDependency(EnumU u, S * source, EnumO i);
  template<typename U = T, typename EnumU, typename S, typename EnumO, typename ... Args>
  void addInputDependency(EnumU u, S* source, EnumO i, Args ... args);
  template<typename U, typename EnumO, typename S, typename EnumI>
  void checkAddDirectDependency(EnumO o);
  template<typename U = T, typename EnumO, typename S, typename EnumI>
  void addDirectDependency(EnumO o, S * source, EnumI i);
};

} // namespace abstract

} // namespace graph

} // namespace tvm

#include "Node.hpp"
