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

#include "Outputs.h"

#include <map>
#include <memory>
#include <set>

namespace tvm
{

class CallGraph;

namespace data
{

/** Inputs store a list of Outputs and the required output signals */
class TVM_DLLAPI Inputs
{
public:
  friend class tvm::CallGraph;

  virtual ~Inputs() = default;

  /** Type used to store the inputs */
  using inputs_t = std::map<std::shared_ptr<Outputs>, std::set<int>>;

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
  template<typename T, typename ... Args>
  void addInput(std::shared_ptr<T> source, Args ... args);
  /** Remove all outputs from a given Output object */
  template<typename T>
  void removeInput(T* source);
  /** Remove outputs from a given Output object */
  template<typename T, typename ... Args>
  void removeInput(T* source, Args ... args);
  /** Retrieve an input from a given Output object */
  template<typename T>
  Iterator getInput(T* source);
  template<typename T>
  Iterator getInput(const std::shared_ptr<T>& source);
private:
  inputs_t inputs_;
};

} // namespace data

} // namespace tvm

#include "Inputs.hpp"
