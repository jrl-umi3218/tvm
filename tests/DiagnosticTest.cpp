/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#include "SolverTestFunctions.h"

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/constraint/internal/LinearizedTaskConstraint.h>
#include <tvm/diagnostic/GraphProbe.h>
#include <tvm/diagnostic/matrix.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/solver/defaultLeastSquareSolver.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/Proportional.h>
#include <tvm/task_dynamics/VelocityDamper.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm::diagnostic;
using namespace Eigen;
using namespace tvm;
using namespace tvm::requirements;
using std::make_shared;

// test that \c isInMatrix applied to S returns the correct value for all elements of M.
template<typename T, typename U>
void testIsInMatrix(T & M, U & S)
{
  M.setZero();
  S.setOnes();
  for(int i = 0; i < M.rows(); ++i)
  {
    for(int j = 0; j < M.cols(); ++j)
    {
      FAST_CHECK_EQ(isInMatrix(M, i, j, S), M(i, j) == 1);
    }
  }
}

TEST_CASE("isInMatrix")
{
  int m = 12, n = 10;
  MatrixXd M(m, n);

  auto B = M.block(2, 3, 7, 6);
  testIsInMatrix(M, B);

  auto C = B.block(1, 1, 3, 3);
  testIsInMatrix(B, C);

  auto D = Map<MatrixXd, 0, Stride<-1, -1>>(M.data() + 1, 4, 5, Stride<-1, -1>(24, 3));
  testIsInMatrix(M, D);

  Matrix3d N;
  testIsInMatrix(M, N);
}

TEST_CASE("GraphProbe")
{
  // The goal of this test is to make sure things compile in a representative use-case (IK problem)
  // and to show what is possible with GraphProbe.
  // It is more an integration test than a unit test.
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  x << Vector2d(0.5, 0.5);

  Space s2(3);
  VariablePtr q = s2.createVariable("q");
  q->value(Vector3d(0.4, -0.6, -0.1));

  auto sf = make_shared<SphereFunction>(x, Vector2d(0, 0), 1);
  auto rf = make_shared<Simple2dRobotEE>(q, Vector2d(-3, 0), Vector3d(1, 1, 1));
  auto idx = make_shared<function::IdentityFunction>(x);
  auto df = make_shared<Difference>(rf, idx);

  VectorXd v = Vector2d::Zero();
  Vector3d b = Vector3d::Constant(1.57);

  LinearizedControlProblem lpb;
  auto t1 = lpb.add(sf == 0., task_dynamics::P(2), {PriorityLevel(0)});
  auto t2 = lpb.add(df == v, task_dynamics::P(2), {PriorityLevel(0)});
  auto t3 = lpb.add(-b <= q <= b, task_dynamics::VelocityDamper({1, 0.01, 0, 0.1}), {PriorityLevel(0)});
  auto t4 = lpb.add(dot(q) == 0., task_dynamics::None(), {PriorityLevel(1), AnisotropicWeight(Vector3d(10, 2, 1))});

  scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions().verbose(true));
  solver.solve(lpb);

  GraphProbe gp;
  gp.registerTVMFunction<SphereFunction>();
  gp.registerTVMFunction<Simple2dRobotEE>();
  gp.registerTVMFunction<Difference>();

  auto l = tvm::graph::internal::Logger::logger().log();

  {
    auto val = gp.listOutputVal(&lpb.updateGraph());
    FAST_CHECK_EQ(val.size(), 32);
    const auto & sphereValue = val[29];
    FAST_CHECK_EQ(std::get<GraphProbe::Output>(sphereValue).name, "Value");
    FAST_CHECK_EQ(std::get<GraphProbe::Output>(sphereValue).owner.type, typeid(SphereFunction));
    FAST_CHECK_EQ(std::get<VariablePtr>(sphereValue), nullptr);
    FAST_CHECK_EQ(std::get<MatrixXd>(sphereValue)(0, 0), -0.5);
  }

  {
    auto val = gp.listOutputVal(l.outputs_[0]);
    FAST_CHECK_EQ(val.size(), 1);
    const auto & sphereValue = val.front();
    FAST_CHECK_EQ(std::get<GraphProbe::Output>(sphereValue).name, "Value");
    FAST_CHECK_EQ(std::get<GraphProbe::Output>(sphereValue).owner.type, typeid(SphereFunction));
    FAST_CHECK_EQ(std::get<VariablePtr>(sphereValue), nullptr);
    FAST_CHECK_EQ(std::get<MatrixXd>(sphereValue)(0, 0), -0.5);
  }

  {
    auto t = gp.followUp(&lpb.updateGraph(), diagnostic::GraphProbe::inRange(2, 3));
    FAST_CHECK_EQ(t.size(), 1);
    const auto & jac = mpark::get<GraphProbe::Output>(t.front()->val);
    FAST_CHECK_EQ(jac.name, "Jacobian");
    FAST_CHECK_EQ(jac.owner.type, typeid(tvm::constraint::internal::LinearizedTaskConstraint));
    FAST_CHECK_EQ(t.front()->children.size(), 1);
  }
  {
    auto t = gp.followUp(l.outputs_[29]);
    const auto & jac = mpark::get<GraphProbe::Output>(t->val);
    FAST_CHECK_EQ(jac.name, "U");
    FAST_CHECK_EQ(jac.owner.type, typeid(tvm::constraint::internal::LinearizedTaskConstraint));
    FAST_CHECK_EQ(t->children.size(), 1);
  }
}
