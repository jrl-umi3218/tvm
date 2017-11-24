#include <tvm/Task.h>
#include <tvm/Variable.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/task_dynamics/Proportional.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/utils/ProtoTask.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace Eigen;
using namespace tvm;
using namespace tvm::utils;


template<typename T>
void checkSimpleBoundOnFunction(const T& bound)
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  auto f = std::make_shared<function::IdentityFunction>(x);
  task_dynamics::P td(2);

  {
    Task t(f == bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::EQUAL);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound == f, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::EQUAL);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(f >= bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::GREATER_THAN);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound <= f, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::GREATER_THAN);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(f <= bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::LOWER_THAN);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound >= f, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::LOWER_THAN);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }
}


template<typename L, typename U>
void checkDoubleBoundsOnFunction(const L& l, const U& u)
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  auto f = std::make_shared<function::IdentityFunction>(x);
  task_dynamics::P td(2);

  {
    Task t(l <= f <= u, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::DOUBLE_SIDED);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY(t.secondBoundTaskDynamics());
  }

  {
    Task t(l >= f >= u, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::DOUBLE_SIDED);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY(t.secondBoundTaskDynamics());
  }
}

template<typename T>
void checkSimpleBoundOnVariable(const T& bound)
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  task_dynamics::P td(2);

  {
    Task t(x == bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::EQUAL);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound == x, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::EQUAL);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(x >= bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::GREATER_THAN);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound <= x, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::GREATER_THAN);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(x <= bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::LOWER_THAN);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound >= x, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::LOWER_THAN);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }
}


template<typename L, typename U>
void checkDoubleBoundsOnVariable(const L& l, const U& u)
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  task_dynamics::P td(2);

  {
    Task t(l <= x <= u, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::DOUBLE_SIDED);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY(t.secondBoundTaskDynamics());
  }

  {
    Task t(l >= x >= u, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::DOUBLE_SIDED);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY(t.secondBoundTaskDynamics());
  }
}

TEST_CASE("Test Proto task")
{
  // scalar-defined bounds
  checkSimpleBoundOnFunction(0.);
  checkDoubleBoundsOnFunction(-1., 3.);
  checkSimpleBoundOnVariable(0.);
  checkDoubleBoundsOnVariable(-1., 3.);

  // vector-defined bounds
  VectorXd l = -Vector2d::Ones();
  VectorXd u = Vector2d::Ones();
  checkSimpleBoundOnFunction(l);
  checkDoubleBoundsOnFunction(l, u);
  checkSimpleBoundOnVariable(l);
  checkDoubleBoundsOnVariable(l, u);

  // mix
  checkDoubleBoundsOnFunction(-1., u);
  checkDoubleBoundsOnFunction(l, 3.);
  checkDoubleBoundsOnVariable(-1., u);
  checkDoubleBoundsOnVariable(l, 3.);

  // expression
  Matrix2d A = Matrix2d::Identity();
  checkSimpleBoundOnFunction(-l);
  checkSimpleBoundOnFunction(A*l - u);
  checkSimpleBoundOnVariable(-l);
  checkSimpleBoundOnVariable(A*l - u);
}
