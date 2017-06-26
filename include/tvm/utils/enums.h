#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace tvm
{

namespace utils
{

/** Constexpr function to compute the size of __VA_ARGS__ in a macro.
 *
 * Does not work if __VA_ARGS__ arguments contains a string with an embedded
 * ',' character. This should not happen in the context we use this function.
 *
 * Inspired by https://stackoverflow.com/a/35868609
 *
 * Counts the number of ',' in the stringified version of __VA_ARGS__
 *
 */
template<unsigned int N>
constexpr unsigned int count_va_args(const char(&s)[N], unsigned int i = 0, unsigned int ret = 0)
{
    return s[i] == '\0' ? ( i == 0 ? 0 : ret + 1 ) : count_va_args(s, i + 1, ret + (s[i] == ','));
}
/** Macro call to count_va_args */
#define COUNT_VA_ARGS(...) tvm::utils::count_va_args(""#__VA_ARGS__)

// Check the macro expansion
static_assert(COUNT_VA_ARGS() == 0, "COUNT_VA_ARGS failed for 0 argument.");
static_assert(COUNT_VA_ARGS(1) == 1, "COUNT_VA_ARGS failed for 1 argument.");
static_assert(COUNT_VA_ARGS(1,2,3) == 3, "COUNT_VA_ARGS failed for 3 arguments.");

#define EXTEND_ENUM(SelfT, ParentId, BaseId, EnumName, Enum0, ...)\
  using ParentId = SelfT::BaseId; \
  enum class EnumName { Enum0 = ParentId::EnumName##Size, ## __VA_ARGS__ };\
  static const unsigned int EnumName##Size = ParentId::EnumName##Size + 1 + COUNT_VA_ARGS(__VA_ARGS__); \
  using BaseId = SelfT;


} // namespace utils

} // namespace tvm
