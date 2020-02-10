/* Copyright 2017-2019 CNRS-AIST JRL and CNRS-UM LIRMM
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
namespace internal
{
  template<template<typename...> class Base>
  struct is_base
  {
    template<typename... T>
    static std::true_type check(Base<T...> const volatile&);
    static std::false_type check(...);
  };

  template <typename T, template<typename...> class Base>
  constexpr bool derives_from() {
    return decltype(is_base<Base>::check(std::declval<const T&>()))::value;
  }

  template <typename T, typename Base>
  constexpr bool derives_from() {
    return std::is_base_of_v<Base, T>;
  }

  /** Used to enable a function for a list of types.
    *
    * To have a function work for T equal or deriving from B1, B2, ..., Bk where
    * Bi are types. use as
    * template<typename T, enable_for_t<T,B1, B2, ..., ..., Bk>=0>
    */
  template<typename T, typename... Base>
  using enable_for_t = std::enable_if_t<(... || (std::is_same_v<T, Base> || derives_from<T, Base>())), int>;


  /** Used to enable a function for a list of templated classes.
    *
    * To have a function work for T equal or deriving from B1, B2, ..., Bk where
    * Bi are a templated classes. use as
    * template<typename T, enable_for_templated_t<T,B1, B2, ..., ..., Bk>=0>
    */
  template<typename T, template<typename...> class... Base>
  using enable_for_templated_t = std::enable_if_t<(... || derives_from<T, Base>()), int>;
}
}
