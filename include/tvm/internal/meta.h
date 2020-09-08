/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <type_traits>

/** For a name \a XXX, this creates a templated class has_member_type_XXX in the
 * namespace tvm::internal such that has_member_type_XXX<T>::value is true if
 * t::XXX is a valid expression and refers to a type, and false otherwise.
 *
 * Adapted from https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Member_Detector
 * (section Detecting member types):
 *
 * > An overload set of two test functions is then created, just like the other
 * > examples. The first version can only be instantiated if the type of U::Type
 * > can be used unambiguously. This type can be used only if there is exactly
 * > one instance of Type in Derived, i.e. there is no Type in T. If T has a
 * > member type Type, it is garanteed to be different than Fallback::Type,
 * > since the latter is a unique type, hence creating the ambiguity. This leads
 * > to the second version of test being instantiated in case the substitution
 * > fails, meaning that T has indeed a member type Type.
 *
 * To avoid the code from failling if Type is not a class, we add an additional
 * step, were T is kept as is if it is a class and replace by an empty \a Dummy
 * class otherwise.
 */
#define TVM_CREATE_HAS_MEMBER_TYPE_TRAIT_FOR(Type)                                       \
  namespace tvm::internal                                                                \
  {                                                                                      \
  template<typename T>                                                                   \
  class has_member_type_##Type                                                           \
  {                                                                                      \
  private:                                                                               \
    struct Dummy                                                                         \
    {};                                                                                  \
    struct Fallback                                                                      \
    {                                                                                    \
      struct Type                                                                        \
      {};                                                                                \
    };                                                                                   \
    struct Derived : std::conditional<std::is_class<T>::value, T, Dummy>::type, Fallback \
    {};                                                                                  \
                                                                                         \
    template<class U>                                                                    \
    static std::false_type test(typename U::Type *);                                     \
    template<typename U>                                                                 \
    static std::true_type test(U *);                                                     \
                                                                                         \
  public:                                                                                \
    static constexpr bool value = decltype(test<Derived>(nullptr))::value;               \
  };                                                                                     \
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

/** A sink, whose value is always true. */
template<typename T>
class always_true : public std::true_type
{};

/** A sink, whose value is always false. */
template<typename T>
class always_false : public std::false_type
{};
} // namespace internal
} // namespace tvm
