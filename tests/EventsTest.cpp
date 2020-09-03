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

#ifdef TVM_USE_LSSOL
#define IF_USE_LSSOL(x) x
#else
#define IF_USE_LSSOL(x) do {} while(0)
#endif
#ifdef TVM_USE_QLD
#define IF_USE_QLD(x) x
#else
#define IF_USE_QLD(x) do {} while(0)
#endif
#ifdef TVM_USE_QUADPROG
#define IF_USE_QUADPROG(x) x
#else
#define IF_USE_QUADPROG(x) do {} while(0)
#endif

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

// Skip if more than one type of constraints/objective is added more than once
bool skip(const std::array<bool, 12>& a)
{
  int eq = a[0] + a[1] + a[2];
  int ineq = a[3] + a[4] + a[5];
  int bnd = a[6] + a[7] + a[8];
  int obj = a[9] + a[10] + a[11];

  int n = (eq > 1) + (ineq > 1) + (bnd > 1) + (obj > 1);
  return n > 1;
}

// Skip if more than one type of constraints/objective is added more than once
// and more than three types is added. 
bool skip2(const std::array<bool, 8>& a)
{
  int eq = a[0] + a[1];
  int ineq = a[2] + a[3];
  int bnd = a[4] + a[5];
  int obj = a[6] + a[7];

  int n1 = (eq > 1) + (ineq > 1) + (bnd > 1) + (obj > 1);
  int n2 = (eq > 0) + (ineq > 0) + (bnd > 0) + (obj > 0);
  return n1 > 1 && n2 > 3;
}

template<int N>
void buildPb(LinearizedControlProblem& pb,
             const std::array<TaskWithRequirementsPtr, N>& tasks,
             const std::array<bool, N>& selection)
{
  for (int i = 0; i < N; ++i)
  {
    if (selection[i]) pb.add(tasks[i]);
  }
}

template<int N>
void checkSolution(const std::array<TaskWithRequirementsPtr, N>& tasks,
                   const std::array<bool, N>& added, VariableVector& x0, 
                   const VectorXd& s0, 
                   VariableVector& x, 
                   const VectorXd s)
{
  Vector2d eps = Vector2d::Constant(1e-6);
  for (int j = 0; j < (3*N)/4; ++j) // all constraints
  {
    if (added[j])
    {
      auto t = tasks[j]->task;
      auto f = std::static_pointer_cast<tvm::function::abstract::LinearFunction>(t.function());
      x.value(s);
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
  for (int j = (3 * N) / 4; j < N; ++j) //objectives
  {
    if (added[j])
    {
      auto t = tasks[j]->task;
      auto f = std::static_pointer_cast<tvm::function::abstract::LinearFunction>(t.function());
      x0.value(s0);
      f->updateValue();
      Vector2d obj0 = f->value();
      x.value(s);
      f->updateValue();
      Vector2d obj = f->value();
      FAST_CHECK_UNARY((obj0 - obj).isZero(1e-6));
    }
  }
}

void test1Change(const std::array<bool, 12>& selection)
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

  IF_USE_LSSOL(scheme::WeightedLeastSquares solverLssol(solver::LSSOLLSSolverOptions{}));
  IF_USE_QLD(scheme::WeightedLeastSquares solverQLD(solver::QLDLSSolverOptions().cholesky(true)));
  IF_USE_QUADPROG(scheme::WeightedLeastSquares solverQuadprog(solver::QuadprogLSSolverOptions().cholesky(true)));

  for (int i = 0; i < 12; ++i)
  {
    std::array<bool, 12> added = selection;
    LinearizedControlProblem pb;
    buildPb(pb, tasks, added);

    // We solve pb for the current list of tasks
    IF_USE_LSSOL(solverLssol.solve(pb));
    IF_USE_QLD(solverQLD.solve(pb));
    IF_USE_QUADPROG(solverQuadprog.solve(pb));
    
    if (added[i])
    {
      pb.remove(tasks[i].get());
      added[i] = false;
    }
    else
    {
      pb.add(tasks[i]);
      added[i] = true;
    }

    // Create a gound truth problem
    LinearizedControlProblem pbGroundTruth;
    buildPb(pbGroundTruth, tasks, added);

    // Now we solve both problem. Only pb is performing an update
#if defined(TVM_USE_LSSOL)
    FAST_CHECK_UNARY(solverLssol.solve(pbGroundTruth));
#elif defined(TVM_USE_QLD)
    FAST_CHECK_UNARY(solverQLD.solve(pbGroundTruth));
#elif defined(TVM_USE_QUADPROG)
    FAST_CHECK_UNARY(solverQuadprog.solve(pbGroundTruth));
#endif
    VectorXd s0 = pbGroundTruth.variables().value();

#ifdef TVM_USE_LSSOL
    FAST_CHECK_UNARY(solverLssol.solve(pb));
    checkSolution(tasks, added, pbGroundTruth.variables(), s0, pb.variables(), pb.variables().value());
#endif
#ifdef TVM_USE_QLD
    FAST_CHECK_UNARY(solverQLD.solve(pb));
    checkSolution(tasks, added, pbGroundTruth.variables(), s0, pb.variables(), pb.variables().value());
#endif
#ifdef TVM_USE_QUADPROG
    FAST_CHECK_UNARY(solverQuadprog.solve(pb));
    checkSolution(tasks, added, pbGroundTruth.variables(), s0, pb.variables(), pb.variables().value());
#endif
  }
}

void test3Change(const std::array<bool, 8>& selection)
{
  Space R2(2);
  VariablePtr x = R2.createVariable("x");
  VariablePtr y = R2.createVariable("y");
  VariablePtr z = R2.createVariable("z");

  using task_dynamics::None;
  SolvingRequirements P0 = { PriorityLevel(0) };
  SolvingRequirements P1 = { PriorityLevel(1) };
  // 3 of each: equality, inequality, bound and objective, in that order
  std::array<TaskWithRequirementsPtr, 8> tasks = {
  std::make_shared<TaskWithRequirements>(Task{ x + 2 * y == 3., None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ 3 * x + z == 4., None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ x + y >= 0.,     None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ x + z <= 3.,     None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ 0. <= x <= 4.,   None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ -2. <= x <= 2.,  None() }, P0),
  std::make_shared<TaskWithRequirements>(Task{ y == 0.,         None() }, P1),
  std::make_shared<TaskWithRequirements>(Task{ z == 0.,         None() }, P1) };

  IF_USE_LSSOL(scheme::WeightedLeastSquares solverLssol(solver::LSSOLLSSolverOptions{}));
  IF_USE_QLD(scheme::WeightedLeastSquares solverQLD(solver::QLDLSSolverOptions().cholesky(true)));
  IF_USE_QUADPROG(scheme::WeightedLeastSquares solverQuadprog(solver::QuadprogLSSolverOptions().cholesky(true)));

  for (int i = 0; i < 8; ++i)
  {
    for (int j = 0; j < 8; ++j)
    {
      for (int k = 0; k < 8; k+=2)
      {
        std::array<bool, 8> added = selection;
        LinearizedControlProblem pb;
        buildPb(pb, tasks, added);

        // We solve pb for the current list of tasks
        IF_USE_LSSOL(solverLssol.solve(pb));
        IF_USE_QLD(solverQLD.solve(pb));
        IF_USE_QUADPROG(solverQuadprog.solve(pb));

        if (added[i])
        {
          pb.remove(tasks[i].get());
          added[i] = false;
        }
        else
        {
          pb.add(tasks[i]);
          added[i] = true;
        }

        if (added[j])
        {
          pb.remove(tasks[j].get());
          added[j] = false;
        }
        else
        {
          pb.add(tasks[j]);
          added[j] = true;
        }

        if (added[k])
        {
          pb.remove(tasks[k].get());
          added[k] = false;
        }
        else
        {
          pb.add(tasks[k]);
          added[k] = true;
        }

        // Create a gound truth problem
        LinearizedControlProblem pbGroundTruth;
        buildPb(pbGroundTruth, tasks, added);

        // Now we solve both problem. Only pb is performing an update
#if defined(TVM_USE_LSSOL)
        FAST_CHECK_UNARY(solverLssol.solve(pbGroundTruth));
#elif defined(TVM_USE_QLD)
        FAST_CHECK_UNARY(solverQLD.solve(pbGroundTruth));
#elif defined(TVM_USE_QUADPROG)
        FAST_CHECK_UNARY(solverQuadprog.solve(pbGroundTruth));
#endif
        VectorXd s0 = pbGroundTruth.variables().value();

#ifdef TVM_USE_LSSOL
        FAST_CHECK_UNARY(solverLssol.solve(pb));
        checkSolution(tasks, added, pbGroundTruth.variables(), s0, pb.variables(), pb.variables().value());
#endif
#ifdef TVM_USE_QLD
        FAST_CHECK_UNARY(solverQLD.solve(pb));
        checkSolution(tasks, added, pbGroundTruth.variables(), s0, pb.variables(), pb.variables().value());
#endif
#ifdef TVM_USE_QUADPROG
        FAST_CHECK_UNARY(solverQuadprog.solve(pb));
        checkSolution(tasks, added, pbGroundTruth.variables(), s0, pb.variables(), pb.variables().value());
#endif
      }
    }
  }
}


TEST_CASE("Systematic add/remove of one task")
{
  // This test is creating a large number of problems, leading to a large log.
  // So we disable the logs here.
  tvm::graph::internal::Logger::logger().disable();

  std::array<bool, 12> added = { false, false, false, false, false, false, false, false, false, false, false, false };
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
      test1Change(added);
    }
  }
}

TEST_CASE("Systematic add/remove of three tasks")
{
  // This test is creating a large number of problems, leading to a large log.
  // So we disable the logs here.
  tvm::graph::internal::Logger::logger().disable();

  std::array<bool, 8> added = { false, false, false, false, false, false, false, false};
  // all combinations of true/false for added
  for (int i = 1; i < 256; ++i)
  {
    // Take the next value of added
    for (int j = 0; j < 8; ++j)
    {
      added[j] = !added[j];
      if (added[j]) break;
    }

    if (!skip2(added))
    {
      test3Change(added);
    }
  }
}
