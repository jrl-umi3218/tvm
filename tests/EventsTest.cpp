/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/function/abstract/LinearFunction.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/solver/defaultLeastSquareSolver.h>

using namespace tvm;
using namespace Eigen;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace Eigen;
using namespace tvm;
using namespace tvm::requirements;

//TEST_CASE("Weight change")
//{
//  Space R2(2);
//  VariablePtr x = R2.createVariable("x");
//
//  LinearizedControlProblem pb;
//  auto t1 = pb.add(x == -1., { PriorityLevel(1) });
//  auto t2 = pb.add(x == Vector2d(1,3), {PriorityLevel(1), Weight(1)});
//
//  scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});
//
//  solver.solve(pb);
//  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0,1)));
//
//  t2->requirements.weight() = 3;
//  solver.solve(pb);
//  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0.5, 2)));
//
//  t1->requirements.weight() = 3;
//  solver.solve(pb);
//  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 1)));
//
//  t1->requirements.anisotropicWeight() = Vector2d(3, 1);
//  t2->requirements.anisotropicWeight() = Vector2d(1, 3);
//  solver.solve(pb);
//  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(-0.5, 2)));
//}

TEST_CASE("Simple add/Remove constraint")
{
  Space R2(2);
  VariablePtr x = R2.createVariable("x");

  LinearizedControlProblem pb;
  auto t1 = pb.add(x == -1., { PriorityLevel(1) });
  auto t2 = pb.add(x == Vector2d(1, 3), { PriorityLevel(1), Weight(1) });

  scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});

  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 1)));

  pb.remove(t1.get());
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(1, 3)));

  pb.add(t1);
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 1)));

  pb.add(x == 0., { PriorityLevel(1) });
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 2./3)));

  pb.remove(t1.get());
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(1./2, 3./2)));

  pb.add(t1);
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 2. / 3)));

}

TEST_CASE("Add/Remove variables from problem")
{
  Space R2(2);
  VariablePtr x = R2.createVariable("x");
  Space R3(3);
  VariablePtr y = R3.createVariable("y");

  LinearizedControlProblem pb;
  auto t1 = pb.add(x == 0., { PriorityLevel(1) });

  scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});

  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 0)));

  auto t2 = pb.add(y == 0., { PriorityLevel(1) });
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 0)));
  FAST_CHECK_UNARY(y->value().isApprox(Vector3d(0, 0, 0)));

  pb.remove(t1.get());
  solver.solve(pb);
  FAST_CHECK_UNARY(y->value().isApprox(Vector3d(0, 0, 0)));
}