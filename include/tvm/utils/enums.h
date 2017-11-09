#pragma once
// Disable ISO C99 requires rest arguments to be used
#pragma GCC system_header

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

/** PP_NARG achieves the same as COUNT_VA_ARGS but it does
 * not work with empty arguments. However, it performs text
 * substitution which we will need where it is used */
#define PP_NARG(...) \
           PP_ID(PP_NARG_(__VA_ARGS__,PP_RSEQ_N()))
#define PP_NARG_(...) \
           PP_ID(PP_ARG_N(__VA_ARGS__))
#define PP_ARG_N( \
  _1, _2, _3, _4, _5, _6, _7, _8, _9,_10,  \
  _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
  _21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
  _31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
  _41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
  _51,_52,_53,_54,_55,_56,_57,_58,_59,_60, \
  _61,_62,_63,_64,N,...) N
#define PP_RSEQ_N() \
  64,63,62,61,60,                   \
  59,58,57,56,55,54,53,52,51,50, \
  49,48,47,46,45,44,43,42,41,40, \
  39,38,37,36,35,34,33,32,31,30, \
  29,28,27,26,25,24,23,22,21,20, \
  19,18,17,16,15,14,13,12,11,10, \
  9,8,7,6,5,4,3,2,1,0

#define PP_MAP(macro, data, ...) \
  PP_ID(PP_APPLY(PP_CHOOSE_MAP_START, PP_NARG(__VA_ARGS__)) \
          (macro, data, __VA_ARGS__))

#define PP_CHOOSE_MAP_START(count) PP_MAP##count

#define PP_APPLY(macro, ...) PP_ID(macro(__VA_ARGS__))

#define PP_MAP1(m, d, x) m(d, x)
#define PP_MAP2(m, d, x, ...) m(d, x) PP_ID(PP_MAP1(m, d, __VA_ARGS__))
#define PP_MAP3(m, d, x, ...) m(d, x) PP_ID(PP_MAP2(m, d, __VA_ARGS__))
#define PP_MAP4(m, d, x, ...) m(d, x) PP_ID(PP_MAP3(m, d, __VA_ARGS__))
#define PP_MAP5(m, d, x, ...) m(d, x) PP_ID(PP_MAP4(m, d, __VA_ARGS__))
#define PP_MAP6(m, d, x, ...) m(d, x) PP_ID(PP_MAP5(m, d, __VA_ARGS__))
#define PP_MAP7(m, d, x, ...) m(d, x) PP_ID(PP_MAP6(m, d, __VA_ARGS__))
#define PP_MAP8(m, d, x, ...) m(d, x) PP_ID(PP_MAP7(m, d, __VA_ARGS__))
#define PP_MAP9(m, d, x, ...) m(d, x) PP_ID(PP_MAP8(m, d, __VA_ARGS__))
#define PP_MAP10(m, d, x, ...) m(d, x) PP_ID(PP_MAP9(m, d, __VA_ARGS__)
#define PP_MAP11(m, d, x, ...) m(d, x) PP_ID(PP_MAP10(m, d, __VA_ARGS__))
#define PP_MAP12(m, d, x, ...) m(d, x) PP_ID(PP_MAP11(m, d, __VA_ARGS__))
#define PP_MAP13(m, d, x, ...) m(d, x) PP_ID(PP_MAP12(m, d, __VA_ARGS__))
#define PP_MAP14(m, d, x, ...) m(d, x) PP_ID(PP_MAP13(m, d, __VA_ARGS__))
#define PP_MAP15(m, d, x, ...) m(d, x) PP_ID(PP_MAP14(m, d, __VA_ARGS__))
#define PP_MAP16(m, d, x, ...) m(d, x) PP_ID(PP_MAP15(m, d, __VA_ARGS__))
#define PP_MAP17(m, d, x, ...) m(d, x) PP_ID(PP_MAP16(m, d, __VA_ARGS__))
#define PP_MAP18(m, d, x, ...) m(d, x) PP_ID(PP_MAP17(m, d, __VA_ARGS__))
#define PP_MAP19(m, d, x, ...) m(d, x) PP_ID(PP_MAP18(m, d, __VA_ARGS__))
#define PP_MAP20(m, d, x, ...) m(d, x) PP_ID(PP_MAP19(m, d, __VA_ARGS__))
#define PP_MAP21(m, d, x, ...) m(d, x) PP_ID(PP_MAP20(m, d, __VA_ARGS__))
#define PP_MAP22(m, d, x, ...) m(d, x) PP_ID(PP_MAP21(m, d, __VA_ARGS__))
#define PP_MAP23(m, d, x, ...) m(d, x) PP_ID(PP_MAP22(m, d, __VA_ARGS__))
#define PP_MAP24(m, d, x, ...) m(d, x) PP_ID(PP_MAP23(m, d, __VA_ARGS__))
#define PP_MAP25(m, d, x, ...) m(d, x) PP_ID(PP_MAP24(m, d, __VA_ARGS__))
#define PP_MAP26(m, d, x, ...) m(d, x) PP_ID(PP_MAP25(m, d, __VA_ARGS__))
#define PP_MAP27(m, d, x, ...) m(d, x) PP_ID(PP_MAP26(m, d, __VA_ARGS__))
#define PP_MAP28(m, d, x, ...) m(d, x) PP_ID(PP_MAP27(m, d, __VA_ARGS__))
#define PP_MAP29(m, d, x, ...) m(d, x) PP_ID(PP_MAP28(m, d, __VA_ARGS__))
#define PP_MAP30(m, d, x, ...) m(d, x) PP_ID(PP_MAP29(m, d, __VA_ARGS__))
#define PP_MAP31(m, d, x, ...) m(d, x) PP_ID(PP_MAP30(m, d, __VA_ARGS__))
#define PP_MAP32(m, d, x, ...) m(d, x) PP_ID(PP_MAP31(m, d, __VA_ARGS__))
#define PP_MAP33(m, d, x, ...) m(d, x) PP_ID(PP_MAP32(m, d, __VA_ARGS__))
#define PP_MAP34(m, d, x, ...) m(d, x) PP_ID(PP_MAP33(m, d, __VA_ARGS__))
#define PP_MAP35(m, d, x, ...) m(d, x) PP_ID(PP_MAP34(m, d, __VA_ARGS__))
#define PP_MAP36(m, d, x, ...) m(d, x) PP_ID(PP_MAP35(m, d, __VA_ARGS__))
#define PP_MAP37(m, d, x, ...) m(d, x) PP_ID(PP_MAP36(m, d, __VA_ARGS__))
#define PP_MAP38(m, d, x, ...) m(d, x) PP_ID(PP_MAP37(m, d, __VA_ARGS__))
#define PP_MAP39(m, d, x, ...) m(d, x) PP_ID(PP_MAP38(m, d, __VA_ARGS__))
#define PP_MAP40(m, d, x, ...) m(d, x) PP_ID(PP_MAP39(m, d, __VA_ARGS__))
#define PP_MAP41(m, d, x, ...) m(d, x) PP_ID(PP_MAP40(m, d, __VA_ARGS__))
#define PP_MAP42(m, d, x, ...) m(d, x) PP_ID(PP_MAP41(m, d, __VA_ARGS__))
#define PP_MAP43(m, d, x, ...) m(d, x) PP_ID(PP_MAP42(m, d, __VA_ARGS__))
#define PP_MAP44(m, d, x, ...) m(d, x) PP_ID(PP_MAP43(m, d, __VA_ARGS__))
#define PP_MAP45(m, d, x, ...) m(d, x) PP_ID(PP_MAP44(m, d, __VA_ARGS__))
#define PP_MAP46(m, d, x, ...) m(d, x) PP_ID(PP_MAP45(m, d, __VA_ARGS__))
#define PP_MAP47(m, d, x, ...) m(d, x) PP_ID(PP_MAP46(m, d, __VA_ARGS__))
#define PP_MAP48(m, d, x, ...) m(d, x) PP_ID(PP_MAP47(m, d, __VA_ARGS__))
#define PP_MAP49(m, d, x, ...) m(d, x) PP_ID(PP_MAP48(m, d, __VA_ARGS__))
#define PP_MAP50(m, d, x, ...) m(d, x) PP_ID(PP_MAP49(m, d, __VA_ARGS__))
#define PP_MAP51(m, d, x, ...) m(d, x) PP_ID(PP_MAP50(m, d, __VA_ARGS__))
#define PP_MAP52(m, d, x, ...) m(d, x) PP_ID(PP_MAP51(m, d, __VA_ARGS__))
#define PP_MAP53(m, d, x, ...) m(d, x) PP_ID(PP_MAP52(m, d, __VA_ARGS__))
#define PP_MAP54(m, d, x, ...) m(d, x) PP_ID(PP_MAP53(m, d, __VA_ARGS__))
#define PP_MAP55(m, d, x, ...) m(d, x) PP_ID(PP_MAP54(m, d, __VA_ARGS__))
#define PP_MAP56(m, d, x, ...) m(d, x) PP_ID(PP_MAP55(m, d, __VA_ARGS__))
#define PP_MAP57(m, d, x, ...) m(d, x) PP_ID(PP_MAP56(m, d, __VA_ARGS__))
#define PP_MAP58(m, d, x, ...) m(d, x) PP_ID(PP_MAP57(m, d, __VA_ARGS__))
#define PP_MAP59(m, d, x, ...) m(d, x) PP_ID(PP_MAP58(m, d, __VA_ARGS__))
#define PP_MAP60(m, d, x, ...) m(d, x) PP_ID(PP_MAP59(m, d, __VA_ARGS__))
#define PP_MAP61(m, d, x, ...) m(d, x) PP_ID(PP_MAP60(m, d, __VA_ARGS__))
#define PP_MAP62(m, d, x, ...) m(d, x) PP_ID(PP_MAP61(m, d, __VA_ARGS__))
#define PP_MAP63(m, d, x, ...) m(d, x) PP_ID(PP_MAP62(m, d, __VA_ARGS__))
#define PP_MAP64(m, d, x, ...) m(d, x) PP_ID(PP_MAP63(m, d, __VA_ARGS__))

#define PP_ID(x) x

#define ENUM_NAME(EnumT, name) \
  v == EnumT::name ? #name : 

#define DECLARE_ENUM(EnumName, Enum0, ...) \
  enum class EnumName { Enum0 = EnumName##Parent::EnumName##Size, ## __VA_ARGS__ };

#define EXTEND_ENUM(EnumName, SelfT, ...)\
  using EnumName##Parent = SelfT::EnumName##Base; \
  PP_ID(DECLARE_ENUM(EnumName, __VA_ARGS__))\
  static constexpr unsigned int EnumName##Size = EnumName##Parent::EnumName##Size + PP_ID(COUNT_VA_ARGS(__VA_ARGS__)); \
  using EnumName##Parent::EnumName##Name; \
  static constexpr const char * EnumName##Name(EnumName v) { return PP_ID(PP_MAP(ENUM_NAME, EnumName, __VA_ARGS__)) "INVALID"; } \
  using EnumName##Base = SelfT; \
  static constexpr auto EnumName##BaseName = #SelfT;

#define DISABLE_SIGNALS(EnumName, ...)\
  bool is##EnumName##StaticallyEnabled(int v) const override { return PP_ID(PP_MAP(DISABLE_SIGNALS_BODY, EnumName, __VA_ARGS__)) true; } \
  template<typename EnumT> \
  static constexpr bool EnumName##StaticallyEnabled(EnumT v) { return PP_ID(PP_MAP(DISABLE_SIGNALS_BODY, EnumName, __VA_ARGS__)) true; }

#define DISABLE_SIGNALS_BODY(EnumT, name) \
  static_cast<int>(v) == static_cast<int>(name) ? false : 

#define CLEAR_DISABLED_SIGNALS(EnumName) \
  bool is##EnumName##StaticallyEnabled(int) const override { return true; }\
  template<typename EnumT> \
  static constexpr bool EnumName##StaticallyEnabled(EnumT) { return true; }

} // namespace utils

} // namespace tvm
