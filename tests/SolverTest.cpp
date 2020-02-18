#include "SolverTestFunctions.h"

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/hint/Substitution.h>
#include <tvm/function/abstract/LinearFunction.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/solver/LSSOLLeastSquareSolver.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/VelocityDamper.h>

using namespace tvm;
using namespace Eigen;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

TEST_CASE("Basic problem")
{

}

TEST_CASE("Substitution")
{
  Space s1(2);
  int dim = 3;
  Space s2(dim);

  VectorXd ddx0;
  VectorXd ddq0;
  {
    VariablePtr x = s1.createVariable("x");
    VariablePtr dx = dot(x);
    x->value(Vector2d(0.5, 0.5));
    dx->value(Vector2d::Zero());

    VariablePtr q = s2.createVariable("q");
    VariablePtr dq = dot(q);
    q->value(Vector3d(0.4, -0.6, 0.9));
    dq->value(Vector3d::Zero());

    auto sf = std::make_shared<SphereFunction>(x, Vector2d(0, 0), 1);
    auto rf = std::make_shared<Simple2dRobotEE>(q, Vector2d(2, 0), Vector3d(1, 1, 1));
    auto idx = std::make_shared<function::IdentityFunction>(x);
    auto df = std::make_shared<Difference>(rf, idx);
    auto idq = std::make_shared<function::IdentityFunction>(dot(q,2));

    VectorXd v(2); v << 0, 0;
    Vector3d b = Vector3d::Constant(1.5);

    double dt = 1e-1;
    LinearizedControlProblem lpb;
    auto t1 = lpb.add(sf == 0., task_dynamics::PD(2), { requirements::PriorityLevel(0) });
    auto t2 = lpb.add(df == v, task_dynamics::PD(2), { requirements::PriorityLevel(0) });
    auto t3 = lpb.add(-b <= q <= b, task_dynamics::VelocityDamper(dt, { 1., 0.01, 0, 1 }), { requirements::PriorityLevel(0) });
    auto t4 = lpb.add(idq == 0., task_dynamics::None(), { requirements::PriorityLevel(1) });

    scheme::WeightedLeastSquares solver(solver::LSSOLLSSolverFactory{});
    solver.solve(lpb);
    ddx0 = dot(x, 2)->value();
    ddq0 = dot(q, 2)->value();
  }

  VectorXd ddxs;
  VectorXd ddqs;
  {
    VariablePtr x = s1.createVariable("x");
    VariablePtr dx = dot(x);
    x->value(Vector2d(0.5, 0.5));
    dx->value(Vector2d::Zero());

    VariablePtr q = s2.createVariable("q");
    VariablePtr dq = dot(q);
    q->value(Vector3d(0.4, -0.6, 0.9));
    dq->value(Vector3d::Zero());

    auto sf = std::make_shared<SphereFunction>(x, Vector2d(0, 0), 1);
    auto rf = std::make_shared<Simple2dRobotEE>(q, Vector2d(2, 0), Vector3d(1, 1, 1));
    auto idx = std::make_shared<function::IdentityFunction>(x);
    auto df = std::make_shared<Difference>(rf, idx);
    auto idq = std::make_shared<function::IdentityFunction>(dot(q, 2));

    VectorXd v(2); v << 0, 0;
    Vector3d b = Vector3d::Constant(1.5);

    double dt = 1e-1;
    LinearizedControlProblem lpb;
    auto t1 = lpb.add(sf == 0., task_dynamics::PD(2), { requirements::PriorityLevel(0) });
    auto t2 = lpb.add(df == v, task_dynamics::PD(2), { requirements::PriorityLevel(0) });
    auto t3 = lpb.add(-b <= q <= b, task_dynamics::VelocityDamper(dt, { 1., 0.01, 0, 1 }), { requirements::PriorityLevel(0) });
    auto t4 = lpb.add(idq == 0., task_dynamics::None(), { requirements::PriorityLevel(1) });

    lpb.add(hint::Substitution(lpb.constraint(t2.get()), dot(x, 2)));

    scheme::WeightedLeastSquares solver(solver::LSSOLLSSolverOptions{});
    solver.solve(lpb);
    ddxs = dot(x, 2)->value();
    ddqs = dot(q, 2)->value();
  }

  FAST_CHECK_UNARY(ddx0.isApprox(ddxs, 1e-10));
  FAST_CHECK_UNARY(ddq0.isApprox(ddqs, 1e-10));
}
