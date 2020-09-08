// Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM

#include <iostream>

#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/solver/defaultLeastSquareSolver.h>
#include <tvm/task_dynamics/Proportional.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/VelocityDamper.h>

#include "SolverTestFunctions.h"

using namespace Eigen;
using namespace tvm;
using namespace tvm::requirements;
using namespace tvm::task_dynamics;
using std::make_shared;

constexpr int maxIt = 1000;
constexpr double dt = 0.1;

VectorXd IKExample()
{
  // Creating variable x in R^2 and initialize it to [0.5, 0.5]
  Space R2(2);
  VariablePtr x = R2.createVariable("x");
  x << 0.5, 0.5;

  // Creating variable q in R^3 and initialize it to [0.4, -0.6, -0.1]
  Space R3(3);
  VariablePtr q = R3.createVariable("q");
  q->value(Vector3d(0.4, -0.6, -0.1));

  // Creating the functions
  auto g = make_shared<Simple2dRobotEE>(q, Vector2d(-3, 0), Vector3d(1, 1, 1)); // g(q)
  auto idx = make_shared<function::IdentityFunction>(x);                        // I x
  auto e1 = make_shared<Difference>(g, idx);                                    // e_1(q,x) = g(q) - x
  auto e2 = make_shared<SphereFunction>(x, Vector2d(0, 0), 1);                  // e_2(x)

  Vector3d b = Vector3d::Constant(tvm::constant::pi / 2);

  // Creating the problem
  ControlProblem pb;
  auto t1 = pb.add(e1 == 0., Proportional(2), PriorityLevel(0));
  auto t2 = pb.add(e2 == 0., Proportional(2), PriorityLevel(0));
  auto t3 = pb.add(-b <= q <= b, VelocityDamper({1, 0.01, 0, 0.1}), PriorityLevel(0));
  auto t4 = pb.add(dot(q) == 0., {PriorityLevel(1), AnisotropicWeight(Vector3d(10, 2, 1))});

  // Linearization
  LinearizedControlProblem lpb(pb);

  // Creating the resolution scheme
  scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});

  // IK loop
  int i = 0;
  do
  {
    solver.solve(lpb);
    x->value(x->value() + dot(x)->value() * dt);
    q->value(q->value() + dot(q)->value() * dt);
    ++i;
  } while((dot(q)->value().norm() > 1e-8 || dot(x)->value().norm() > 1e-8) && i < maxIt);

  std::cout << "At q = " << q->value().transpose() << ",\n    e1 = " << e1->value().transpose() << "\n"
            << "   convergence in " << i << " iterations" << std::endl;

  return q->value();
}

VectorXd IKSubstitutionExample()
{
  Space R2(2);
  VariablePtr x = R2.createVariable("x");
  x << 0.5, 0.5;

  Space R3(3);
  VariablePtr q = R3.createVariable("q");
  q->value(Vector3d(0.4, -0.6, -0.1));

  auto g = make_shared<Simple2dRobotEE>(q, Vector2d(-3, 0), Vector3d(1, 1, 1)); // g(q)
  auto idx = make_shared<function::IdentityFunction>(x);                        // I x
  auto e1 = make_shared<Difference>(g, idx);                                    // e_1(q,x) = g(q) - x
  auto e2 = make_shared<SphereFunction>(x, Vector2d(0, 0), 1);                  // e_2(x)

  Vector3d b = Vector3d::Constant(tvm::constant::pi / 2);

  ControlProblem pb;
  auto t1 = pb.add(e1 == 0., Proportional(2), PriorityLevel(0));
  auto t2 = pb.add(e2 == 0., Proportional(2), PriorityLevel(0));
  auto t3 = pb.add(-b <= q <= b, VelocityDamper({1, 0.01, 0, 0.1}), PriorityLevel(0));
  auto t4 = pb.add(dot(q) == 0., {PriorityLevel(1), AnisotropicWeight(Vector3d(10, 2, 1))});

  LinearizedControlProblem lpb(pb);
  lpb.add(hint::Substitution(lpb.constraint(t1.get()), dot(x)));

  scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});
  int i = 0;
  do
  {
    solver.solve(lpb);
    x->value(x->value() + dot(x)->value() * dt);
    q->value(q->value() + dot(q)->value() * dt);
    ++i;
  } while((dot(q)->value().norm() > 1e-8 || dot(x)->value().norm() > 1e-8) && i < maxIt);

  std::cout << "At q = " << q->value().transpose() << ",\n    e1 = " << e1->value().transpose() << "\n"
            << "   convergence in " << i << " iterations" << std::endl;

  return q->value();
}

// Let's run some quick tests to ensure this example is not outdated and compiles.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

TEST_CASE("Runing IK examples")
{
  VectorXd q1 = IKExample();
  VectorXd q2 = IKSubstitutionExample();
  FAST_CHECK_UNARY(q1.isApprox(q2, 1e-6));
}