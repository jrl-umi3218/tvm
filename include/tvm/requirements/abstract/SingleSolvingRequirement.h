/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

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
  const T & value() const { return value_; }

  /** Change the current value */
  void value(const T & val)
  {
    value_ = val;
    default_ = false;
    if constexpr(!Lightweight)
    {
      Base::run();
    }
  }

  /** check it the requirement is at its default value. */
  bool isDefault() const { return default_; }

protected:
  SingleSolvingRequirement(const T & val, bool isDefault) : default_(isDefault), value_(val) {}

  SingleSolvingRequirement(const SingleSolvingRequirement<T, !Lightweight> & other)
  : value_(other.value()), default_(other.isDefault())
  {
  }

  SingleSolvingRequirement & operator=(const SingleSolvingRequirement<T, !Lightweight> & other)
  {
    value_ = other.value();
    default_ = other.isDefault();
    return *this;
  }

  SingleSolvingRequirement & operator=(const T & val)
  {
    value(val);
    return *this;
  }

  /** Is this requirement at its default value?*/
  bool default_;

  T value_;
};

} // namespace abstract

} // namespace requirements

} // namespace tvm

#define TVM_DEFINE_LW_NON_LW_CONVERSION_OPERATORS(className, T, L)                            \
  className(const className<!L> & other) : abstract::SingleSolvingRequirement<T, L>(other) {} \
  className & operator=(const className<!L> & other)                                          \
  {                                                                                           \
    abstract::SingleSolvingRequirement<T, L>::operator=(other);                               \
    return *this;                                                                             \
  }                                                                                           \
  className & operator=(const T & val)                                                        \
  {                                                                                           \
    abstract::SingleSolvingRequirement<T, L>::operator=(val);                                 \
    return *this;                                                                             \
  }
