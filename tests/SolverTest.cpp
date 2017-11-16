#include "SolverTestFunctions.h"

#include <iostream>

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/function/abstract/LinearFunction.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/VelocityDamper.h>

using namespace tvm;
using namespace Eigen;

void solverTest01()
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  VariablePtr dx = dot(x);
  x->value(Vector2d(0.5,0.5));
  dx->value(Vector2d::Zero());

  int dim = 3;
  Space s2(dim);
  VariablePtr q = s2.createVariable("q");
  VariablePtr dq = dot(q);
  q->value(Vector3d(0.4, -0.6, 0.9));
  dq->value(Vector3d::Zero());

  auto sf = std::make_shared<SphereFunction>(x, Vector2d(0, 0), 1);
  auto rf = std::make_shared<Simple2dRobotEE>(q, Vector2d(2, 0), Vector3d(1, 1, 1));
  auto idx = std::make_shared<function::IdentityFunction>(x);
  auto df = std::make_shared<Difference>(rf, idx);

  VectorXd v(2); v << 0, 0;
  Vector3d b = Vector3d::Constant(1.5);

  double dt = 1e-1;
  LinearizedControlProblem lpb;
  auto t1 = lpb.add(sf == 0., task_dynamics::PD(2), { requirements::PriorityLevel(0) });
  auto t2 = lpb.add(df == v, task_dynamics::PD(2), { requirements::PriorityLevel(0) });
  auto t3 = lpb.add(-b <= q <= b, task_dynamics::VelocityDamper(dt, { 1., 0.01, 0, 1 }), { requirements::PriorityLevel(0) });
  std::cout << t1->task.taskDynamics<task_dynamics::PD>()->gains().first << std::endl;

  scheme::WeightedLeastSquares solver;
  solver.solve(lpb);
  std::cout << "ddx = " << dot(x, 2)->value().transpose() << std::endl;
  std::cout << "ddq = " << dot(q, 2)->value().transpose() << std::endl;
}


void solverTest02()
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");

  Space s2(3);
  VariablePtr q = s2.createVariable("q");

  auto idx = std::make_shared<function::IdentityFunction>(x);
  auto idq = std::make_shared<function::IdentityFunction>(q);

  ControlProblem pb;
  pb.add(idx >= 0., task_dynamics::None(), { requirements::PriorityLevel(0) });
  pb.add(idq >= 0., task_dynamics::None(), { requirements::PriorityLevel(0) });

  LinearizedControlProblem lpb(pb);

  scheme::WeightedLeastSquares solver;
  solver.solve(lpb);
}
