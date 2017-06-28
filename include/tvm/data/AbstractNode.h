#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "../api.h"

#include "Inputs.h"

#include <cassert>

namespace tvm
{

struct CallGraph;

namespace data
{

/** An abstract node is the structure stored in the call graph to hide away the
 * actual type of the node.
 *
 */
struct AbstractNode : public Inputs, Outputs
{
  template<typename T>
  friend struct Node;

  friend struct tvm::CallGraph;

  /** Base Update enumeration. Empty */
  enum class Update {};

  /** Store the size of the Update enumeration */
  static constexpr unsigned int UpdateSize = 0;

  /** Meta-information regarding the class inheritance diagram */
  using UpdateParent = AbstractNode;

  /** Meta-information used during inheritance to retrieve the base-class output size */
  using UpdateBase = AbstractNode;

  /** Meta-information holding the name of the class to which the Update enum belong */
  static constexpr auto UpdateBaseName = "AbstractNode";

  /** Return the name of a given update */
  static constexpr const char * UpdateName(Output) { return "INVALID"; }

  /** Check if a given update is enabled (run-time)
   *
   * The default implementation always return true. The
   * expected parameter is int to swallow the different
   * update types.
   *
   */
  virtual bool isUpdateEnabled(int) { return true; }

  /** Check if a given update is enabled (compile-time)
   *
   * The default implementation always return true.
   *
   */
  template<typename EnumT>
  static constexpr bool UpdateEnabled(EnumT) { return true; }

  virtual ~AbstractNode() = default;

  /** Call the function stored at index i */
  inline void update(int i)
  {
    assert(updates_.count(i));
    updates_[i]();
  }
protected:
  std::map<int, std::function<void()>> updates_;

  std::map<int, std::vector<int>> outputDependencies_;
  std::map<int, std::vector<int>> internalDependencies_;
  using input_dependency_t = std::map<std::shared_ptr<Outputs>, std::set<int>>;
  std::map<int, input_dependency_t> inputDependencies_;
private:
  AbstractNode()
  {
    is_node_ = true;
  }
};

/** Add new update signals for a given entity SelfT.
 *
 * Read the meta-information available at construction for SelfT to start the
 * enum value at the correct value.
 *
 * A Node object that does not need new update signals does not need to
 * call this macro.
 *
 */
#define SET_UPDATES(SelfT, ...)\
  EXTEND_ENUM(Update, SelfT, __VA_ARGS__)

/** Mark some update signals as disabled for that class */
#define DISABLE_UPDATES(...)\
  DISABLE_SIGNALS(Update, __VA_ARGS__)

/** Check if a value of EnumT is a valid update for Node type T */
template<typename T, typename EnumT>
constexpr bool is_valid_update(EnumT v)
{
  static_assert(std::is_base_of<AbstractNode, T>::value, "Cannot test update validity for a type that is not derived of Updates");
  static_assert(std::is_enum<EnumT>::value, "Cannot test update validity for a value that is not an enumeration");
  return std::is_same<typename T::Update, EnumT>::value ||
         ( (!std::is_same<typename T::UpdateParent, typename T::UpdateBase>::value) &&
           is_valid_update<typename T::UpdateParent>(v) );
}

/** Check if all values of a given set are valid updates for Updates type T */
template<typename T, typename EnumT, typename ... Args>
constexpr bool is_valid_update(EnumT v, Args ... args)
{
  return is_valid_update<T>(v) && is_valid_update<T>(args...);
}

} // namespace data

} // namespace tvm
