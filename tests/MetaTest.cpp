/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/internal/meta.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <Eigen/Core>

#include <tvm/utils/AffineExpr.h>

using namespace tvm::internal;
using namespace tvm::utils;


//---------------------- derives_from ------------------------\\

static_assert(derives_from<Eigen::MatrixXd, Eigen::MatrixBase>());
static_assert(!derives_from<double, Eigen::MatrixBase>());
static_assert(derives_from<LinearExpr<Eigen::MatrixXd>, LinearExpr>());
static_assert(!derives_from<Eigen::MatrixXd, LinearExpr>());
static_assert(derives_from<AffineExpr<::internal::NoConstant, Eigen::MatrixXd>, AffineExpr>());
static_assert(!derives_from<Eigen::MatrixXd, AffineExpr>());
static_assert(derives_from<Eigen::MatrixXd, Eigen::MatrixXd>());
static_assert(!derives_from<int, int>()); //derives_from only work with classes

//---------- enable_for_t and enable_for_templated_t ---------\\

// Dummy classes for test purposes
class A {};
template<typename U, typename V> class TemplatedClass {};
template<typename U> class TemplatedClassD1 : public TemplatedClass<U, int> {};
class TemplatedClassD2 : public TemplatedClass<double, int> {};
class TemplatedClassD3 : public TemplatedClassD1<int> {};

//function accepting int and Eigen::MatrixXd
template<typename T, enable_for_t<T, int, Eigen::MatrixXd> = 0>
constexpr std::true_type testSimple(const T&) { return {}; }
// Fallback version
constexpr std::false_type testSimple(...) { return {}; }

//function accepting Eigen::MatrixBase, tvm::utils::LinearExpr, tvm::utils::AffineExpr
template<typename T, enable_for_templated_t<T, Eigen::MatrixBase, TemplatedClass> = 0>
constexpr std::true_type testTemplate(const T&) { return {}; }
// Fallback version
constexpr std::false_type testTemplate(...) { return {}; }

static_assert(testSimple(3));
static_assert(!testSimple(6.));
static_assert(decltype(testSimple(Eigen::MatrixXd()))::value);
static_assert(!decltype(testSimple(Eigen::Matrix4d()))::value);
static_assert(!decltype(testSimple(A()))::value);

static_assert(decltype(testTemplate(Eigen::MatrixXd()))::value);
static_assert(decltype(testTemplate(Eigen::Matrix4d()))::value);
static_assert(!decltype(testTemplate(A()))::value);
static_assert(decltype(testTemplate(TemplatedClass<int, double>()))::value);
static_assert(decltype(testTemplate(TemplatedClassD1<int>()))::value);
static_assert(decltype(testTemplate(TemplatedClassD2()))::value);
static_assert(decltype(testTemplate(TemplatedClassD3()))::value);


//---------------- always_true, always_false -----------------\\

static_assert(always_true<int>::value);
static_assert(always_true<A>::value);
static_assert(!always_false<int>::value);
static_assert(!always_false<A>::value);

//------------------ has_member_type_XXX ---------------------\\

TVM_CREATE_HAS_MEMBER_TYPE_TRAIT_FOR(Foo)

class B { using Foo = int; };
static_assert(!has_member_type_Foo<int>::value);
static_assert(!has_member_type_Foo<A>::value);
static_assert(has_member_type_Foo<B>::value);

//---------------------- is_detected -------------------------\\

class C
{
  int i;
  int foo(int k);
public:
  int j;
  int bar(int k);
  int bar(B b, double d);
};

template<typename T>
using i_trait = decltype(std::declval<T>().i);

template<typename T>
using j_trait = decltype(std::declval<T>().j);

template<typename T>
using foo_trait = decltype(std::declval<T>().foo(0));

template<typename T>
using bar_trait = decltype(std::declval<T>().bar(0));

template<typename T, typename U>
using bar2_trait = decltype(std::declval<T>().bar(std::declval<U>(), 3.14));

static_assert(!is_detected<i_trait, A>::value);
#if not defined(MSVC) // There seems to be a regression bug with vc19.25 where the idom ignore private accessibility
static_assert(!is_detected<i_trait, C>::value);
#endif
static_assert(!is_detected<j_trait, A>::value);
static_assert(is_detected<j_trait, C>::value);
static_assert(!is_detected<foo_trait, A>::value);
#if not defined MSVC // There seems to be a regression bug with vc19.25 where the idom ignore private accessibility
static_assert(!is_detected<foo_trait, C>::value);
#endif
static_assert(!is_detected<bar_trait, A>::value);
static_assert(is_detected<bar_trait, C>::value);
static_assert(!is_detected<bar2_trait, A, B>::value);
static_assert(is_detected<bar2_trait, C, B>::value);
static_assert(!is_detected<bar2_trait, C, A>::value);
