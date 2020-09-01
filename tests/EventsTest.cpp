/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/function/abstract/LinearFunction.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/solver/defaultLeastSquareSolver.h>
#ifdef TVM_USE_LSSOL
# include <tvm/solver/LSSOLLeastSquareSolver.h>
#endif
#ifdef TVM_USE_QLD
# include <tvm/solver/QLDLeastSquareSolver.h>
#endif
#ifdef TVM_USE_QUADPROG
# include <tvm/solver/QuadprogLeastSquareSolver.h>
#endif

#include <array>

using namespace tvm;
using namespace Eigen;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace Eigen;
using namespace tvm;
using namespace tvm::requirements;

TEST_CASE("Weight change")
{
  Space R2(2);
  VariablePtr x = R2.createVariable("x");

  LinearizedControlProblem pb;
  auto t1 = pb.add(x == -1., { PriorityLevel(1) });
  auto t2 = pb.add(x == Vector2d(1,3), {PriorityLevel(1), Weight(1)});

  scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});

  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0,1)));

  t2->requirements.weight() = 3;
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0.5, 2)));

  t1->requirements.weight() = 3;
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 1)));

  t1->requirements.anisotropicWeight() = Vector2d(3, 1);
  t2->requirements.anisotropicWeight() = Vector2d(1, 3);
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(-0.5, 2)));
}

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

bool skip(const std::array<bool, 12>& a)
{
  int eq = a[0] + a[1] + a[2];
  int ineq = a[3] + a[4] + a[5];
  int bnd = a[6] + a[7] + a[8];
  int obj = a[9] + a[10] + a[11];

  int n = (eq > 1) + (ineq > 1) + (bnd > 1) + (obj > 1);
  return n > 1;
}

void buildPb(LinearizedControlProblem& pb,
             const std::array<TaskWithRequirementsPtr, 12>& tasks,
             const std::array<bool, 12>& selection)
{
  for (int i = 0; i < 12; ++i)
  {
    if (selection[i]) pb.add(tasks[i]);
  }
}

void test1Change(const std::array<TaskWithRequirementsPtr, 12>& tasks, const std::array<bool, 12>& selection, const scheme::WeightedLeastSquares& solver)
{
  for (int i = 0; i < 12; ++i)
  {
    LinearizedControlProblem pb;
    LinearizedControlProblem pbGroundTruth;
    buildPb(pb, tasks, selection);
    buildPb(pbGroundTruth, tasks, selection);

    // We solve pb for the current list of tasks, but not pbGroundTruth
    solver.solve(pb);
    
    if (selection[i])
    {
      pb.remove(tasks[i].get());
      pbGroundTruth.remove(tasks[i].get());
    }
    else
    {
      pb.add(tasks[i]);
      pbGroundTruth.add(tasks[i]);
    }

    // Now we solve both problem. Only pb is performing an update
    FAST_CHECK_UNARY(solver.solve(pb));
    VectorXd s = pb.variables().value();
    FAST_CHECK_UNARY(solver.solve(pbGroundTruth));
    VectorXd s0 = pbGroundTruth.variables().value();

    Vector2d eps = Vector2d::Constant(1e-6);
    for (int j = 0; j < 9; ++j) // all constraints
    {
      if ((selection[j] && i != j) || (!selection[j] && i == j))
      {
        auto t = tasks[j]->task;
        auto f = std::static_pointer_cast<tvm::function::abstract::LinearFunction>(t.function());
        pb.variables().value(s);
        f->updateValue();
        Vector2d v = f->value() - tasks[j]->task.taskDynamics<task_dynamics::None>()->value();
        switch (t.type())
        {
        case constraint::Type::EQUAL: FAST_CHECK_UNARY(v.isZero(1e-6)); break;
        case constraint::Type::GREATER_THAN: FAST_CHECK_UNARY((v.array() >= -eps.array()).all()); break;
        case constraint::Type::LOWER_THAN: FAST_CHECK_UNARY((v.array() <= eps.array()).all()); break;
        case constraint::Type::DOUBLE_SIDED:
        {
          Vector2d v2 = f->value() - tasks[j]->task.secondBoundTaskDynamics<task_dynamics::None>()->value();
          FAST_CHECK_UNARY((v.array() >= -eps.array()).all());
          FAST_CHECK_UNARY((v2.array() <= eps.array()).all());
        }
        break;
        }
      }
    }
    for (int j = 9; j < 12; ++j) //objectives
    {
      if ((selection[j] && i != j) || (!selection[j] && i == j))
      {
        auto t = tasks[j]->task;
        auto f = std::static_pointer_cast<tvm::function::abstract::LinearFunction>(t.function());
        pbGroundTruth.variables().value(s0);
        f->updateValue();
        Vector2d obj0 = f->value();
        pb.variables().value(s);
        f->updateValue();
        Vector2d obj = f->value();
        FAST_CHECK_UNARY((obj0-obj).isZero(1e-6));
      }
    }
  }

}

TEST_CASE("Systematic add/remove of one task")
{
  Space R2(2);
  VariablePtr x = R2.createVariable("x");
  VariablePtr y = R2.createVariable("y");
  VariablePtr z = R2.createVariable("z");

  using task_dynamics::None;
  SolvingRequirements P0 = { PriorityLevel(0) };
  SolvingRequirements P1 = { PriorityLevel(1) };
  // 3 of each: equality, inequality, bound and objective, in that order
  std::array<TaskWithRequirementsPtr, 12> tasks = {
  std::make_shared<TaskWithRequirements>(Task{ x + 2 * y == 3., None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ 3 * x + z == 4., None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ y - z == 0.,     None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ x + y >= 0.,     None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ x + y + z >= 1., None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ x + z <= 3.,     None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ 0. <= x <= 4.,   None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ -2. <= x <= 2.,  None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ x <= 3.,         None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ x == 0.,         None() }, P1),
  std::make_shared<TaskWithRequirements>(Task{ y == 0.,         None() }, P1),
  std::make_shared<TaskWithRequirements>(Task{ z == 0.,         None() }, P1) };
  std::array<bool, 12> added = {false, false, false, false, false, false, false, false, false, false, false, false};

  // all combinations of true/false for added
  for (int i = 1; i < 4096; ++i)
  {
    // Take the next value of added
    for (int j = 0; j < 12; ++j)
    {
      added[j] = !added[j];
      if (added[j]) break;
    }

    if (!skip(added))
    {
#ifdef TVM_USE_LSSOL
      {
        scheme::WeightedLeastSquares solver(solver::LSSOLLSSolverOptions{});
        test1Change(tasks, added, solver);
      }
#endif
#ifdef TVM_USE_QLD
      {
        scheme::WeightedLeastSquares solver(solver::QLDLSSolverOptions().cholesky(true));
        test1Change(tasks, added, solver);
      }
#endif
#ifdef TVM_USE_QUADPROG
      {
        scheme::WeightedLeastSquares solver(solver::QuadprogLSSolverOptions().cholesky(true));
        test1Change(tasks, added, solver);
      }
#endif
    }
  }
}
