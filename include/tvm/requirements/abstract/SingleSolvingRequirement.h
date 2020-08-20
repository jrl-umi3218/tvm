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

#include <variant>

#include <tvm/internal/CallbackManager.h>

namespace tvm
{

namespace requirements
{

namespace abstract
{

  /** A class representing the way a constraint has to be solved and how it
    * interacts with other constraints in term of hierarchical and weighted
    * priorities.
    *
    * This is a base class for the sole purpose of conveniency.
    */
  template<typename T, bool Lightweight = true>
  class SingleSolvingRequirement : public std::conditional_t<Lightweight, std::monostate, internal::CallbackManager>
  {
    using Base = std::conditional_t<Lightweight, std::monostate, internal::CallbackManager>;

  public:
    /** Get the current value. */
    const T& value() const { return value_; }

    /** Change the current value */
    void value(const T& val)
    {
      value_ = val;
      default_ = false;
      if constexpr (!Lightweight)
      {
        Base::run();
      }
    }

    /** check it the requirement is at its default value. */
    bool isDefault() const { return default_; }

  protected:
    SingleSolvingRequirement(const T& val, bool isDefault)
      : default_(isDefault), value_(val)
    {}

    SingleSolvingRequirement(const SingleSolvingRequirement<T, !Lightweight>& other)
      : value_(other.value()), default_(other.isDefault())
    {}

    SingleSolvingRequirement& operator=(const SingleSolvingRequirement<T, !Lightweight>& other)
    {
      value_ = other.value();
      default_ = other.isDefault();
      return *this;
    }

    SingleSolvingRequirement& operator=(const T& val) { value(val); return *this; }

    /** Is this requirement at its default value?*/
    bool default_;

    T value_;
  };

}  // namespace abstract

}  // namespace requirements

}  // namespace tvm


#define TVM_DEFINE_LW_NON_LW_CONVERSION_OPERATORS(className, T, L)  \
  className(const className<!L>& other)                         \
   : abstract::SingleSolvingRequirement<T,L>(other) {}          \
  className& operator=(const className<!L>& other)              \
  {                                                             \
    abstract::SingleSolvingRequirement<T,L>::operator=(other);  \
    return *this;                                               \
  }                                                             \
  className& operator=(const T& val)                            \
  {                                                             \
    abstract::SingleSolvingRequirement<T, L>::operator=(val);   \
    return *this;                                               \
  }
