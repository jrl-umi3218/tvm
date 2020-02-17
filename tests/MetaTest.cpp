#include <tvm/internal/meta.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <Eigen/Core>

#include <tvm/utils/AffineExpr.h>

using namespace tvm::internal;
using namespace tvm::utils;

static_assert(derives_from<Eigen::MatrixXd, Eigen::MatrixBase>());
static_assert(!derives_from<double, Eigen::MatrixBase>());
static_assert(derives_from<LinearExpr<Eigen::MatrixXd>, LinearExpr>());
static_assert(!derives_from<Eigen::MatrixXd, LinearExpr>());
static_assert(derives_from<AffineExpr<::internal::NoConstant, Eigen::MatrixXd>, AffineExpr>());
static_assert(!derives_from<Eigen::MatrixXd, AffineExpr>());
static_assert(derives_from<Eigen::MatrixXd, Eigen::MatrixXd>());
static_assert(!derives_from<int, int>()); //derives_from only work with classes

//------------------------------------------------------------\\

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