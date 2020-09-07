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

#include <tvm/graph/abstract/Outputs.h>

#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace tvm
{

namespace graph
{

class CallGraph;

namespace internal
{

/** Inputs store a list of Outputs and the required output signals */
class TVM_DLLAPI Inputs
{
public:
  friend class tvm::graph::CallGraph;

  virtual ~Inputs() = default;

  /** Type used to store the inputs */
  using inputs_t = std::unordered_map<abstract::Outputs *, std::set<int>>;
  /** Type used to store the inputs data */
  using store_t = std::unordered_set<std::shared_ptr<abstract::Outputs>>;

  /** A simple extension to inputs_t iterator that also stores the end
   * iterator, allowing to cast the iterator to a boolean value.
   *
   */
  struct TVM_DLLAPI Iterator : public inputs_t::iterator
  {
    /** Construct from an existing iterator and the end iterator */
    Iterator(inputs_t::iterator it, inputs_t::iterator end);
    /** True if the iterator is valid */
    operator bool();

  private:
    inputs_t::iterator end_;
  };

  /** Add outputs from a given Output object */
  template<typename T, typename EnumI, typename... Args>
  void addInput(std::shared_ptr<T> source, EnumI i, Args... args);

  /** Add outputs from a given Output object.
   *
   * Leave the caller responsible for the source lifetime
   */
  template<typename T,
           typename EnumI,
           typename... Args,
           typename std::enable_if<std::is_base_of<abstract::Outputs, T>::value, int>::type = 0>
  void addInput(T & source, EnumI i, Args... args);
  /** Remove all outputs from a given Output object */
  template<typename T>
  void removeInput(T * source);
  /** Remove outputs from a given Output object */
  template<typename T, typename... Args>
  void removeInput(T * source, Args... args);
  /** Retrieve an input from a given Output object */
  template<typename T>
  Iterator getInput(T * source);
  /** Retrieve an input from a given Output object */
  template<typename T>
  Iterator getInput(const std::shared_ptr<T> & source);

private:
  /** Remove an input with a given iterator and source */
  void removeInput(Iterator it, abstract::Outputs * source);
  inputs_t inputs_;
  store_t store_;
  /** Add a single output from a given Output object. */
  template<typename T, typename EnumI>
  void addInput(T * source, EnumI i);
  /** Terminal case */
  template<typename T>
  void addInput(T &)
  {
  }
};

} // namespace internal

} // namespace graph

} // namespace tvm

#include "Inputs.hpp"
