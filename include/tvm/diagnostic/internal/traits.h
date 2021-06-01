/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Variable.h>

namespace tvm::diagnostic::internal
{

/** This structure is specialized to match the methods we want to register */
template<typename T, typename MethodT>
struct CheckAccessor
{
  static constexpr bool isVoidAccessor = false;
  static constexpr bool isVariableAccessor = false;
};

/** Helper to avoid repeating ourselves on matching void methods */
template<typename T>
struct ValidVoidAccessor
{
  using ReturnT = T;
  static constexpr bool isVoidAccessor = true;
  static constexpr bool isVariableAccessor = false;
};

/** Helper to avoid repeating ourselves on matching variables methods */
template<typename T>
struct ValidVariableAccessor
{
  using ReturnT = T;
  static constexpr bool isVoidAccessor = false;
  static constexpr bool isVariableAccessor = true;
};

template<typename T, typename U>
struct CheckAccessor<T, U (T::*)() const> : public ValidVoidAccessor<U>
{};

template<typename T, typename U>
struct CheckAccessor<T, U (T::*)() const noexcept> : public ValidVoidAccessor<U>
{};

#ifndef _MSC_VER
template<typename T, typename U, typename Base>
struct CheckAccessor<T, U (Base::*)() const> : public ValidVoidAccessor<U>
{
  static_assert(std::is_base_of_v<Base, T>, "This method does not belong to T hierarchy");
};

template<typename T, typename U, typename Base>
struct CheckAccessor<T, U (Base::*)() const noexcept> : public ValidVoidAccessor<U>
{
  static_assert(std::is_base_of_v<Base, T>, "This method does not belong to T hierarchy");
};
#endif

template<typename T, typename U>
struct CheckAccessor<T, U (T::*)(const Variable &) const> : public ValidVariableAccessor<U>
{};

template<typename T, typename U>
struct CheckAccessor<T, U (T::*)(const Variable &) const noexcept> : public ValidVariableAccessor<U>
{};

#ifndef _MSC_VER
template<typename T, typename U, typename Base>
struct CheckAccessor<T, U (Base::*)(const Variable &) const> : public ValidVariableAccessor<U>
{
  static_assert(std::is_base_of_v<Base, T>, "This method does not belong to T hierarchy");
};

template<typename T, typename U, typename Base>
struct CheckAccessor<T, U (Base::*)(const Variable &) const noexcept> : public ValidVariableAccessor<U>
{
  static_assert(std::is_base_of_v<Base, T>, "This method does not belong to T hierarchy");
};
#endif

/** Given a return type and an argument type returns a convert function if possible */
template<typename ArgT, typename ConvertT>
std::function<Eigen::MatrixXd(const ArgT &)> MakeConvert(ConvertT && convertIn)
{
  using RetT = std::function<Eigen::MatrixXd(const ArgT &)>;
  if constexpr(std::is_constructible_v<RetT, ConvertT &&>)
  {
    return RetT(convertIn);
  }
  else
  {
    return [](const ArgT & u) { return u; };
  }
}

} // namespace tvm::diagnostic::internal
