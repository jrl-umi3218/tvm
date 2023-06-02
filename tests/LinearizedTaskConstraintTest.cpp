/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include "SolverTestFunctions.h"

#include <tvm/Variable.h>
#include <tvm/constraint/internal/LinearizedTaskConstraint.h>
#include <tvm/task_dynamics/Proportional.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/utils/ProtoTask.h>
#include <tvm/utils/graph.h>

using namespace Eigen;
using namespace tvm;
using LTC = tvm::constraint::internal::LinearizedTaskConstraint;

TEST_CASE("Construction Test")
{
  VariablePtr x = Space(3).createVariable("x");
  auto dx = dot(x);
  auto ddx = dot(dx);
  auto f = std::make_shared<SphereFunction>(x, Vector3d(1, 0, 0), 2);

  task_dynamics::P td1(2);
  task_dynamics::PD td2(2, 1);

  LTC l1(f == 0., td1);
  FAST_CHECK_EQ(l1.size(), 1);
  FAST_CHECK_EQ(l1.type(), constraint::Type::EQUAL);
  FAST_CHECK_EQ(l1.rhs(), constraint::RHS::AS_GIVEN);
  FAST_CHECK_EQ(l1.variables()[0], dx);
  FAST_CHECK_UNARY(l1.linearIn(*dx));

  LTC l2(f <= 0., td2);
  FAST_CHECK_EQ(l2.size(), 1);
  FAST_CHECK_EQ(l2.type(), constraint::Type::LOWER_THAN);
  FAST_CHECK_EQ(l2.rhs(), constraint::RHS::AS_GIVEN);
  FAST_CHECK_EQ(l2.variables()[0], ddx);
  FAST_CHECK_UNARY(l2.linearIn(*ddx));

  LTC l3(-1. <= f, td2);
  FAST_CHECK_EQ(l3.type(), constraint::Type::GREATER_THAN);

  LTC l4(-1. <= f <= 1., td2);
  FAST_CHECK_EQ(l4.type(), constraint::Type::DOUBLE_SIDED);
}

TEST_CASE("Value test")
{
  VariablePtr x = Space(3).createVariable("x");
  auto dx = dot(x);
  auto ddx = dot(dx);

  x << 1, 2, 3;
  dx << -1, -2, -3;

  auto f = std::make_shared<SphereFunction>(x, Vector3d(1, 0, 0), 2);
  task_dynamics::P td1(2);
  task_dynamics::PD td2(2, 1);

  auto l1 = std::make_shared<LTC>(f == 0., td1);
  auto l2 = std::make_shared<LTC>(f <= 0., td2);

  auto E = LTC::Output::E;
  auto U = LTC::Output::U;
  auto J = LTC::Output::Jacobian;
  auto graph1 = utils::generateUpdateGraph(l1, E, J);
  graph1->execute();
  CHECK_EQ(l1->e()[0], -2 * 9);
  CHECK_UNARY(l1->jacobian(*dx).isApprox(Vector3d(0, 4, 6).transpose()));

  auto graph2 = utils::generateUpdateGraph(l2, U, J);
  graph2->execute();
  CHECK_EQ(l2->u()[0], -2 * 9 - (-26) - 28); /*-kp*f - kv*df/dt - d2f/dxdt dx/dt*/
  CHECK_UNARY(l2->jacobian(*ddx).isApprox(Vector3d(0, 4, 6).transpose()));
}
