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

#include "../utils/enums.h"

#include <type_traits>

namespace tvm
{

struct CallGraph;

namespace data
{

/** Outputs simply describe a list of Outputs in a strongly typed enum named
 * Output.
 *
 * An extension mechanism is used to allow extending existing outputs
 * entitites. Note that this prevent multiple-inheritance for Outputs objects
 * (i.e. a new Ouputs must follow a direct inheritance line).
 *
 */
struct TVM_DLLAPI Outputs
{
  friend struct tvm::CallGraph;

  virtual ~Outputs() = default;

  /** Base Output enumeration. Empty */
  enum class Output {};

  /** Store the size of the Output enumeration */
  static constexpr unsigned int OutputSize = 0;

  /** Meta-information regarding the class inheritance diagram */
  using OutputParent = Outputs;

  /** Meta-information used during inheritance to retrieve the base-class output size */
  using OutputBase = Outputs;

  /** Meta-information holding the name of the class to which the Output enum belong */
  static constexpr auto OutputBaseName = "Outputs";

  /** Return the name of a given output */
  static constexpr const char * OutputName(Output) { return "INVALID"; }

  /** Check if a given output is enabled (run-time)
   *
   * The default implementation always return true. The
   * expected parameter is int to swallow the different
   * output types.
   *
   */
  virtual bool isOutputEnabled(int) { return true; }

  /** Check if a given output is enabled (compile-time)
   *
   * The default implementation always return true.
   *
   */
  template<typename EnumT>
  static constexpr bool OutputEnabled(EnumT) { return true; }
protected:
  /** Used to avoid a dynamic cast when working on Outputs that may be tvm::data::Node */
  bool is_node_ = false;
};

/** Add new output signals for a given entity SelfT.
 *
 * Read the meta-information available at construction for SelfT to start the
 * enum value at the correct value.
 *
 * An Outputs object that does not need new output signals does not need to
 * call this macro.
 *
 */
#define SET_OUTPUTS(SelfT, ...)\
  EXTEND_ENUM(Output, SelfT, __VA_ARGS__)

/** Mark some output signals as disabled for that class */
#define DISABLE_OUTPUTS(...)\
  DISABLE_SIGNALS(Output, __VA_ARGS__)

/** Check if a value of EnumT is a valid output for Outputs type T */
template<typename T, typename EnumT>
constexpr bool is_valid_output(EnumT v)
{
  static_assert(std::is_base_of<Outputs, T>::value, "Cannot test output validity for a type that is not derived of Outputs");
  static_assert(std::is_enum<EnumT>::value, "Cannot test output validity for a value that is not an enumeration");
  return std::is_same<typename T::Output, EnumT>::value ||
         ( (!std::is_same<typename T::OutputParent, typename T::OutputBase>::value) &&
           is_valid_output<typename T::OutputParent>(v) );
}

/** Check if all values of a given set are valid outputs for Outputs type T */
template<typename T, typename EnumT, typename ... Args>
constexpr bool is_valid_output(EnumT v, Args ... args)
{
  return is_valid_output<T>(v) && is_valid_output<T>(args...);
}

} // namespace data

} // namespace tvm
