/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include "SolverTestFunctions.h"

#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#ifdef TVM_USE_LSSOL
# include <tvm/solver/LSSOLLeastSquareSolver.h>
#endif
#ifdef TVM_USE_QLD
# include <tvm/solver/QLDLeastSquareSolver.h>
#endif
#ifdef TVM_USE_QUADPROG
# include <tvm/solver/QuadprogLeastSquareSolver.h>
#endif
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/Proportional.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/VelocityDamper.h>

using namespace tvm;
using namespace tvm::requirements;
using namespace tvm::solver;
using namespace tvm::solver::abstract;
using namespace Eigen;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

std::unique_ptr<LinearizedControlProblem> circleIK()
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  x << Vector2d(0.5, 0.5);

  Space s2(3);
  VariablePtr q = s2.createVariable("q");
  q->value(Vector3d(0.4, -0.6, -0.1));

  auto sf = std::make_shared<SphereFunction>(x, Vector2d(0, 0), 1);
  auto rf = std::make_shared<Simple2dRobotEE>(q, Vector2d(-3, 0), Vector3d(1, 1, 1));
  auto idx = std::make_shared<function::IdentityFunction>(x);
  auto damp = std::make_shared<function::IdentityFunction>(dot(q));
  auto df = std::make_shared<Difference>(rf, idx);

  VectorXd v = Vector2d::Zero();
  Vector3d b = Vector3d::Constant(1.57);

  auto lpb = std::make_unique<LinearizedControlProblem>();
  auto t1 = lpb->add(sf == 0., task_dynamics::P(2), { PriorityLevel(0) });
  auto t2 = lpb->add(df == v, task_dynamics::P(2), { PriorityLevel(0) });
  auto t3 = lpb->add(-b <= q <= b, task_dynamics::VelocityDamper({ 1, 0.01, 0, 0.1 }), { PriorityLevel(0) });
  auto t4 = lpb->add(damp == 0., task_dynamics::None(), { PriorityLevel(1), AnisotropicWeight(Vector3d(10,2,1)) });

  return lpb;
}

void testSolvers(const std::unique_ptr<LinearizedControlProblem>& lpb, std::vector<std::shared_ptr<solver::abstract::LSSolverFactory> > configs, double eps)
{
  VariableVector variables = lpb->variables();
  std::vector<VectorXd> solutions;

  for (const auto& c : configs)
  {
    scheme::WeightedLeastSquares s(*c);
    s.solve(*lpb);
    solutions.push_back(variables.value());
  }

  for (size_t i = 0; i < solutions.size(); ++i)
  {
    for (size_t j = i + 1; j < solutions.size(); ++j)
    {
      FAST_CHECK_UNARY(solutions[i].isApprox(solutions[j], eps));
    }
  }
}

TEST_CASE("Simple IK")
{
  auto lpb = circleIK();
  std::vector<std::shared_ptr<LSSolverFactory> > configs;
#ifdef TVM_USE_LSSOL
  configs.push_back(std::make_shared<LSSOLLSSolverFactory>());
#endif
#ifdef TVM_USE_QLD
  configs.push_back(std::make_shared<QLDLSSolverFactory>());
  configs.push_back(std::make_shared<QLDLSSolverFactory>(QLDLSSolverOptions().cholesky(true)));
#endif
#ifdef TVM_USE_QUADPROG
  configs.push_back(std::make_shared<QuadprogLSSolverFactory>());
  configs.push_back(std::make_shared<QuadprogLSSolverFactory>(QuadprogLSSolverOptions().cholesky(true)));
#endif

  testSolvers(lpb, configs, 1e-6);
}
