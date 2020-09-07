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

#include <optional>
#include <type_traits>

/** Adding getter and setter for the option \a optionName of type \a type. */
#define TVM_ADD_OPTION_GET_SET(optionName, type)                           \
public:                                                                    \
  const std::optional<type> & optionName() const { return optionName##_; } \
  auto & optionName(const type & v)                                        \
  {                                                                        \
    optionName##_ = v;                                                     \
    return *this;                                                          \
  }

/** Adding an option \a optionName of type \a type with no default value.
 * (The default value of the underlying solver will be used.
 */
#define TVM_ADD_DEFAULT_OPTION(optionName, type) \
private:                                         \
  using optionName##_t = type;                   \
  std::optional<type> optionName##_;             \
  TVM_ADD_OPTION_GET_SET(optionName, type)

/** Adding an option \a optionName with new default value \a defaultValue, that
 * will replace the value of the underlying solver.
 */
#define TVM_ADD_NON_DEFAULT_OPTION(optionName, defaultValue)          \
private:                                                              \
  using optionName##_t = std::remove_const_t<decltype(defaultValue)>; \
  std::optional<optionName##_t> optionName##_ = defaultValue;         \
  TVM_ADD_OPTION_GET_SET(optionName, optionName##_t)

/** Process \a optionName: if \a optionName has a non-default value, use
 * \a target.setterName to set that value for \a target.
 */
#define TVM_PROCESS_OPTION_2(optionName, target, setterName) \
  if(options.optionName())                                   \
    target.setterName(options.optionName().value());

/** Specialized version of TVM_PROCESS_OPTION_2 where \a setterName = \a optionName. */
#define TVM_PROCESS_OPTION(optionName, target) TVM_PROCESS_OPTION_2(optionName, target, optionName)