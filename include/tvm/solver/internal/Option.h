/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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

#include <type_traits>

namespace tvm
{
namespace solver
{
namespace internal
{
  /** A class representing a variable of type \a T, that can have an additional
    * value \a default, signaled by the fact that \a keepDefault() returns 
    * \a true.
    * It is meant to be use as a solver option where the \a default value means
    * that the solver option should not be changed.
    */
  template<typename T>
  class Option
  {
  public:
    /** Create a variable with the \a default value*/
    Option() : keepDefault_(true), value_() {}
    /** Create a variable with a non-default value*/
    Option(const T& value) : keepDefault_(false), value_(value) {}
    Option(const Option& other) = default;
    Option(Option&& other) = default;

    Option& operator=(const Option& other) = default;
    /** Assign a non-dafault value.*/
    Option& operator=(const T& value) { keepDefault_ = false; value_ = value; return *this; }

    bool keepDefault() const { return keepDefault_; }
    const T& value() const { return value_; }

    void resetToDefault() { keepDefault_ = true; }

  private:
    bool keepDefault_;
    T value_;
  };
} //internal
} //solver
} //tvm

#define ADD_OPTION_GET_SET(optionName, type)                                          \
public:                                                                               \
  const solver::internal::Option<type>& optionName() const { return optionName##_; }  \
  auto& optionName(const type& v) { optionName##_ = v; return *this; }

#define ADD_DEFAULT_OPTION(optionName, type)    \
private:                                        \
  using optionName##_t = type;                  \
  solver::internal::Option<type> optionName##_; \
ADD_OPTION_GET_SET(optionName, type)

#define ADD_NON_DEFAULT_OPTION(optionName, defaultValue)                  \
private:                                                                  \
  using optionName##_t = std::remove_const<decltype(defaultValue)>::type; \
  solver::internal::Option<optionName##_t> optionName##_ = defaultValue;  \
ADD_OPTION_GET_SET(optionName, optionName##_t)

#define PROCESS_OPTION_2(optionName, target, setterName)\
if (!options.optionName().keepDefault())                \
  target.setterName(options.optionName().value());

#define PROCESS_OPTION(optionName, target) PROCESS_OPTION_2(optionName, target, optionName)