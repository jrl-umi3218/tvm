/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <tvm/api.h>
#include <tvm/internal/enums.h>

#include <type_traits>

namespace tvm
{

namespace graph
{

class CallGraph;

namespace abstract
{

/** Outputs simply describe a list of Outputs in a strongly typed enum named
 * Output.
 *
 * An extension mechanism is used to allow extending existing outputs
 * entitites. Note that this prevent multiple-inheritance for Outputs objects
 * (i.e. a new Ouputs must follow a direct inheritance line).
 *
 */
class TVM_DLLAPI Outputs
{
public:
  friend class tvm::graph::CallGraph;

  virtual ~Outputs() = default;

  /** Base Output enumeration. Empty */
  enum class Output_ {};
  /** Base class for Output. Empty */
  struct Output {};

  /** Store the size of the Output enumeration */
  static constexpr unsigned int OutputSize = 0;

  /** Meta-information regarding the class inheritance diagram */
  using OutputParent = Outputs;

  /** Meta-information used during inheritance to retrieve the base-class output size */
  using OutputBase = Outputs;

  /** Meta-information holding the name of the class to which the Output enum belong */
  static constexpr auto OutputBaseName = "Outputs";

  /** Return the name of a given output */
  static constexpr const char * OutputName(Output_) { return "INVALID"; }

  /** Check if a given output is enabled, be it at the class (static) or
   * instance (dynamic) level).
   */
  template <typename EnumT>
  bool isOutputEnabled(EnumT e) const
  {
    // FIXME is there a way to check that the enum has the good type here ?
    int i = static_cast<int>(e);
    return isOutputEnabled(i);
  }

  /** Same as above, but taking int, for conveniency*/
  bool isOutputEnabled(int i) const
  {
    return isOutputStaticallyEnabled(i) && isOutputCustomEnabled(i);
  }

  /** Check if a given output is enabled at the class level (run-time).
  *
  * The default implementation always returns true. The
  * expected parameter is int to swallow the different
  * output types.
  *
  */
  virtual bool isOutputStaticallyEnabled(int) const { return true; }

  /** Check if an output is enabled given a custom criterion
   *
   * The default implementation always returns true.
   * This is a handle for the user to override.
   **/
  virtual bool isOutputCustomEnabled(int) const { return true;}

  /** Check if a given output is enabled at the class level (compile-time)
   *
   * The default implementation always returns true.
   *
   */
  template<typename EnumT>
  static constexpr bool OutputStaticallyEnabled(EnumT) { return true; }

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
  PP_ID(EXTEND_ENUM(Output, SelfT, __VA_ARGS__))

/** Mark some output signals as disabled for that class
  * This overrides Outputs::isOutputEnabled.
  */
#define DISABLE_OUTPUTS(...)\
  PP_ID(DISABLE_SIGNALS(Output, __VA_ARGS__))

/** Mark all outputs as enabled */
#define CLEAR_DISABLED_OUTPUTS()\
  PP_ID(CLEAR_DISABLED_SIGNALS(Output))

/** Check if a value of EnumT is a valid output for Outputs type T */
template<typename T, typename EnumT>
constexpr bool is_valid_output(EnumT v)
{
  static_assert(std::is_base_of<Outputs, T>::value, "Cannot test output validity for a type that is not derived of Outputs");
  static_assert(std::is_enum<EnumT>::value, "Cannot test output validity for a value that is not an enumeration");
  return std::is_same<typename T::Output_, EnumT>::value ||
         ( (!std::is_same<typename T::OutputParent, typename T::OutputBase>::value) &&
           is_valid_output<typename T::OutputParent>(v) );
}

/** Check if all values of a given set are valid outputs for Outputs type T */
template<typename T, typename EnumT, typename ... Args>
constexpr bool is_valid_output(EnumT v, Args ... args)
{
  return is_valid_output<T>(v) && is_valid_output<T>(args...);
}

} // namespace abstract

}  // namespace graph

} // namespace tvm
