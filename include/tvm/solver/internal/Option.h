/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

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
