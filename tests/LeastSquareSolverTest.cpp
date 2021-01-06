/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include "SolverTestFunctions.h"

#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/hint/internal/DiagonalCalculator.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#ifdef TVM_USE_LSSOL
#  include <tvm/solver/LSSOLLeastSquareSolver.h>
#endif
#ifdef TVM_USE_QLD
#  include <tvm/solver/QLDLeastSquareSolver.h>
#endif
#ifdef TVM_USE_QUADPROG
#  include <tvm/solver/QuadprogLeastSquareSolver.h>
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
  auto t1 = lpb->add(sf == 0., task_dynamics::P(2), {PriorityLevel(0)});
  auto t2 = lpb->add(df == v, task_dynamics::P(2), {PriorityLevel(0)});
  auto t3 = lpb->add(-b <= q <= b, task_dynamics::VelocityDamper({1, 0.01, 0, 0.1}), {PriorityLevel(0)});
  auto t4 = lpb->add(damp == 0., task_dynamics::None(), {PriorityLevel(1), AnisotropicWeight(Vector3d(10, 2, 1))});

  return lpb;
}

std::unique_ptr<LinearizedControlProblem> problemWithSubVariables()
{
  VariablePtr x = Space(5).createVariable("x");
  VariablePtr y = Space(3).createVariable("y");

  VariablePtr x1 = x->subvariable(Space(3), "x1");
  VariablePtr x2 = x->subvariable(Space(3), "x2", Space(2));
  VariablePtr y1 = y->subvariable(Space(1), "y1");

  Eigen::VectorXd e = Eigen::VectorXd::Ones(5);

  auto lpb = std::make_unique<LinearizedControlProblem>();
  lpb->add(x1 - y == 0., PriorityLevel(0));
  lpb->add(e.transpose() * x == 0., PriorityLevel(0));
  lpb->add(y == 1., PriorityLevel(1));
  lpb->add(-1 <= x2 <= 0);
  lpb->add(2 <= y1);
  return lpb;
}

std::unique_ptr<LinearizedControlProblem> problemWithSubVariablesAndSubstitution()
{
  VariablePtr q = Space(20).createVariable("q");
  VariablePtr q1 = q->subvariable(6, "q1", 0); // q1 includes q2
  VariablePtr q2 = q->subvariable(1, "q2", 5);
  VariablePtr q3 = q->subvariable(4, "q3", 6);
  VariablePtr q4 = q->subvariable(6, "q4", 10); // q4 includes q5
  VariablePtr q5 = q->subvariable(1, "q5", 15);
  VariablePtr q6 = q->subvariable(4, "q6", 16);

  VectorXd m(4);
  m << 1, -1, 1, -1;

  auto lpb = std::make_unique<LinearizedControlProblem>();
  auto s1 = lpb->add(q3 - m * q2 == 0.);
  auto s2 = lpb->add(q6 - m * q5 == 0.);
  lpb->add(q1 + q4 == 2.);
  lpb->add(q1 - q4 == 0., PriorityLevel(1));
  // This might be a bit surprising, but, because {q2, q3} and {q5, q6} are merged in the constraints,
  // the properties of the matrices in front of q3 and q6 are not kept and the correct calculator
  // cannot be deduced automatically.
  lpb->add(
      hint::Substitution(lpb->constraint(s1.get()), q3, constant::fullRank, tvm::hint::internal::DiagonalCalculator{}));
  lpb->add(
      hint::Substitution(lpb->constraint(s2.get()), q6, constant::fullRank, tvm::hint::internal::DiagonalCalculator{}));

  return lpb;
}

VectorXd testSolvers(const std::unique_ptr<LinearizedControlProblem> & lpb,
                     std::vector<std::shared_ptr<solver::abstract::LSSolverFactory>> configs,
                     double eps)
{
  VariableVector variables = lpb->variables();
  std::vector<VectorXd> solutions;

  for(const auto & c : configs)
  {
    scheme::WeightedLeastSquares s(*c);
    s.solve(*lpb);
    solutions.push_back(variables.value());
    // Solve a second time
    s.solve(*lpb);
    FAST_CHECK_UNARY(solutions.back().isApprox(variables.value()));
  }

  for(size_t i = 0; i < solutions.size(); ++i)
  {
    std::cout << solutions[i].transpose() << std::endl;
    for(size_t j = i + 1; j < solutions.size(); ++j)
    {
      std::cout << solutions[j].transpose() << std::endl;
      FAST_CHECK_UNARY(solutions[i].isApprox(solutions[j], eps));
    }
  }

  return solutions.front();
}

TEST_CASE("Simple IK")
{
  auto lpb = circleIK();
  std::vector<std::shared_ptr<LSSolverFactory>> configs;
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

TEST_CASE("Problem with subvariables")
{
  auto lpb = problemWithSubVariables();
  std::vector<std::shared_ptr<LSSolverFactory>> configs;
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

TEST_CASE("Problem with subvariables and substitutions")
{
  auto lpb = problemWithSubVariablesAndSubstitution();
  std::vector<std::shared_ptr<LSSolverFactory>> configs;
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

  VectorXd x0(20);
  x0 << 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1;
  VectorXd x = testSolvers(lpb, configs, 1e-6);
  FAST_CHECK_UNARY(x.isApprox(x0));
  FAST_CHECK_EQ(typeid(*lpb->substitutions().substitutions()[0].calculator()).hash_code(),
                typeid(tvm::hint::internal::DiagonalCalculator::Impl).hash_code());
  FAST_CHECK_EQ(typeid(*lpb->substitutions().substitutions()[1].calculator()).hash_code(),
                typeid(tvm::hint::internal::DiagonalCalculator::Impl).hash_code());
}
