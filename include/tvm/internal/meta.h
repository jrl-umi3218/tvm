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

/** For a name \a XXX, this creates a templated class has_member_type_XXX such
 * that has_member_type_XXX<T>::value is true if t::XXX is a valid expression
 * and refers to a type, and false otherwise.
 *
 * Adapted from https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Member_Detector
 */
#define CREATE_HAS_MEMBER_TYPE_TRAIT_FOR(Type)                                \
template <typename T>                                                         \
class has_member_type_##Type                                                  \
{                                                                             \
private:                                                                      \
  struct Dummy {};                                                            \
  struct Fallback { struct Type {}; };                                        \
  struct Derived :                                                            \
    std::conditional<std::is_class<T>::value,T,Dummy>::type, Fallback {};     \
                                                                              \
  template<class U>                                                           \
  static std::false_type test(typename U::Type*);                             \
  template<typename U>                                                        \
  static std::true_type test(U*);                                             \
public:                                                                       \
  static constexpr bool value = decltype(test<Derived>(nullptr))::value;      \
};

namespace tvm
{

namespace internal
{
  /** A sink, whose value is always true. */
  template<typename T>
  class always_true : public std::true_type {};

  /** A sink, whose value is always false. */
  template<typename T>
  class always_false : public std::false_type {};
}

}