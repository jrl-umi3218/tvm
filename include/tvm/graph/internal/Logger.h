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
#include <tvm/graph/internal/Log.h>

#include <algorithm>
#include <fstream>
#include <memory>

namespace tvm
{

namespace graph
{
class CallGraph;
namespace abstract
{
class Outputs;
}

namespace internal
{
class Inputs;

class TVM_DLLAPI Logger
{
public:
  static Logger & logger();

  void disable();
  void enable();

  const Log & log() const;

  template<typename U, typename EnumT>
  void registerUpdate(U * node, EnumT u, void (U::*fn)());

  template<typename S, typename EnumO>
  void addInput(Inputs * node, S * source, EnumO i);

  template<typename U, typename EnumO, typename EnumU>
  void addOutputDependency(U * node, EnumO o, EnumU u);

  template<typename U, typename EnumU1, typename EnumU2>
  void addInternalDependency(U * node, EnumU1 uDependent, EnumU2 u);

  template<typename U, typename EnumU, typename S, typename EnumO>
  void addInputDependency(U * node, EnumU u, S * source, EnumO i);

  template<typename U, typename EnumO, typename S, typename EnumI>
  void addDirectDependency(U * node, EnumO o, S * source, EnumI i);

  void addGraphOutput(CallGraph * g, Inputs * node);

  /** Register the type associated to a pointer. */
  template<typename U>
  void registerType(U * node);

  template<typename U>
  void logCall(U * node, void (U::*fn)());

private:
  Logger() = default;

  // raw log
  Log log_;
  bool disabled_;
};

// Helper function for pointer-to-member-function
template<typename T>
std::uintptr_t getPointerValue(void (T::*ptfm)())
{
  auto cptr = reinterpret_cast<std::uintptr_t *>(&ptfm);
  return *cptr;
}

inline const Log & Logger::log() const { return log_; }

template<typename U, typename EnumT>
inline void Logger::registerUpdate(U * node, EnumT u, void (U::*fn)())
{
  if(disabled_)
    return;

  Log::Update up{Log::EnumValue(u), U::UpdateName(u), getPointerValue<U>(fn), Log::Pointer(node)};
  log_.updates_.push_back(up);
  registerType(node);
}

template<typename S, typename EnumO>
inline void Logger::addInput(Inputs * node, S * source, EnumO i)
{
  if(disabled_)
    return;

  Log::Input in = {Log::EnumValue(i), S::OutputName(i), Log::Pointer(source), Log::Pointer(node)};
  log_.inputs_.push_back(in);
  registerType(node);
  registerType(source);
}

template<typename U, typename EnumO, typename EnumU>
inline void Logger::addOutputDependency(U * node, EnumO o, EnumU u)
{
  if(disabled_)
    return;

  Log::Output out = {Log::EnumValue(o), U::OutputName(o), Log::Pointer(node)};
  log_.outputs_.push_back(out);

  Log::OutputDependency dep = {Log::EnumValue(u), Log::EnumValue(o), Log::Pointer(node)};
  log_.outputDependencies_.push_back(dep);
  registerType(node);
}

template<typename U, typename EnumU1, typename EnumU2>
inline void Logger::addInternalDependency(U * node, EnumU1 uDependent, EnumU2 u)
{
  if(disabled_)
    return;

  Log::InternalDependency dep = {Log::EnumValue(u), Log::EnumValue(uDependent), Log::Pointer(node)};
  log_.internalDependencies_.push_back(dep);
  registerType(node);
}

template<typename U, typename EnumU, typename S, typename EnumO>
inline void Logger::addInputDependency(U * node, EnumU u, S * source, EnumO i)
{
  if(disabled_)
    return;

  Log::InputDependency dep = {Log::EnumValue(i), Log::EnumValue(u), Log::Pointer(source), Log::Pointer(node)};
  log_.inputDependencies_.push_back(dep);
  registerType(node);
  registerType(source);
}

template<typename U, typename EnumO, typename S, typename EnumI>
inline void Logger::addDirectDependency(U * node, EnumO o, S * source, EnumI i)
{
  if(disabled_)
    return;

  Log::Output out = {Log::EnumValue(o), U::OutputName(o), Log::Pointer(node)};
  log_.outputs_.push_back(out);

  Log::DirectDependency dep = {Log::EnumValue(i), Log::EnumValue(o), Log::Pointer(source), Log::Pointer(node)};
  log_.directDependencies_.push_back(dep);
  registerType(node);
  registerType(source);
}

template<typename U>
inline void Logger::registerType(U * node)
{
  if(disabled_)
    return;

  std::type_index t(typeid(*node));
  std::uintptr_t val = reinterpret_cast<std::uintptr_t>(node);
  auto & types = log_.types_[val];
  auto it = std::find(types.begin(), types.end(), t);
  if(it == types.end())
  {
    types.push_back(t);
  }
}

} // namespace internal

} // namespace graph

} // namespace tvm
