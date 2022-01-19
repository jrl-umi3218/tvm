/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <type_traits>

/** For a name \a XXX, this creates a templated struct has_public_member_type_XXX in the
 * namespace tvm::internal such that has_public_member_type_XXX<T>::value is true if
 * T::XXX is a public, valid expression and refers to a type, and false otherwise.
 */
#define TVM_CREATE_HAS_MEMBER_TYPE_TRAIT_FOR(Type)                                            \
  namespace tvm::internal                                                                     \
  {                                                                                           \
  struct has_type_##Type##_helper                                                             \
  {                                                                                           \
    template<typename T>                                                                      \
    std::false_type has(...);                                                                 \
                                                                                              \
    template<typename T>                                                                      \
    std::true_type has(typename T::Type *);                                                   \
  };                                                                                          \
                                                                                              \
  template<typename T>                                                                        \
  using has_public_member_type_##Type = decltype(has_type_##Type##_helper{}.has<T>(nullptr)); \
  }

namespace tvm
{
namespace internal
{
/** An helper struct used by derives_from.*/
template<template<typename...> class Base>
struct is_base
{
  /** Accept any class derived from Base<T...>.*/
  template<typename... T>
  static std::true_type check(Base<T...> const volatile &);
  /** Fallback function that will be used for type not deriving from Base<T...>. */
  static std::false_type check(...);
};

/** Check if class \t T derives from the templated class \t Base.
 *
 * This relies on tvm::internal::is_base::check: if T derives from \t Base,
 * the overload returning \a std::true_type will be selected, otherwise, it
 * will be the one returning \a std::false_type.
 *
 * Adapted from https://stackoverflow.com/a/5998303/11611648
 */
template<typename T, template<typename...> class Base>
constexpr bool derives_from()
{
  return decltype(is_base<Base>::check(std::declval<const T &>()))::value;
}

/** Check if class \t T derives from the non-templated class \t Base
 * This returns \a false if \t Base is not a class.
 */
template<typename T, typename Base>
constexpr bool derives_from()
{
  return std::is_base_of_v<Base, T>;
}

/** Used to enable a function for a list of types.
 *
 * To have a function work for T equal or deriving from any B1, B2, ... or Bk
 * where Bi are types.
 * Use as template<typename T, enable_for_t<T,B1, B2, ..., ..., Bk>=0>
 */
template<typename T, typename... Base>
using enable_for_t = std::enable_if_t<(... || (std::is_same_v<T, Base> || derives_from<T, Base>())), int>;

/** Used to enable a function for a list of templated classes.
 *
 * To have a function work for T equal or deriving from any B1, B2, ... or Bk
 * where Bi are a templated classes.
 * Use as template<typename T, enable_for_templated_t<T,B1, B2, ..., ..., Bk>=0>
 */
template<typename T, template<typename...> class... Base>
using enable_for_templated_t = std::enable_if_t<(... || derives_from<T, Base>()), int>;

/** Used to disable a function for a list of templated classes. */
template<typename T, template<typename...> class... Base>
using disable_for_templated_t = std::enable_if_t<!(... || derives_from<T, Base>()), int>;

/** A sink, whose value is always true. */
template<typename T>
class always_true : public std::true_type
{};

/** A sink, whose value is always false. */
template<typename T>
class always_false : public std::false_type
{};

/** Conditionally add \c const to T.*/
template<typename T, bool c>
using const_if = std::conditional<c, const T, T>;

/** C++14-style helper*/
template<typename T, bool c>
using const_if_t = typename const_if<T, c>::type;

} // namespace internal
} // namespace tvm
