/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/function/abstract/LinearFunction.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/solver/defaultLeastSquareSolver.h>
#ifdef TVM_USE_LSSOL
#  include <tvm/solver/LSSOLLeastSquareSolver.h>
#endif
#ifdef TVM_USE_QLD
#  include <tvm/solver/QLDLeastSquareSolver.h>
#endif
#ifdef TVM_USE_QUADPROG
#  include <tvm/solver/QuadprogLeastSquareSolver.h>
#endif

#include <array>
#include <bitset>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#ifdef TVM_USE_LSSOL
#  define IF_USE_LSSOL(x) x
#else
#  define IF_USE_LSSOL(x) \
    do                    \
    {                     \
    } while(0)
#endif
#ifdef TVM_USE_QLD
#  define IF_USE_QLD(x) x
#else
#  define IF_USE_QLD(x) \
    do                  \
    {                   \
    } while(0)
#endif
#ifdef TVM_USE_QUADPROG
#  define IF_USE_QUADPROG(x) x
#else
#  define IF_USE_QUADPROG(x) \
    do                       \
    {                        \
    } while(0)
#endif

#ifdef TVM_USE_JRLQP
#  define IF_USE_JRLQP(x) x
#else
#  define IF_USE_JRLQP(x) \
    do                       \
    {                        \
    } while(0)
#endif

using namespace Eigen;
using namespace tvm;
using namespace tvm::requirements;

TEST_CASE("Weight change")
{
  Space R2(2);
  VariablePtr x = R2.createVariable("x");

  LinearizedControlProblem pb;
  auto t1 = pb.add(x == -1., {PriorityLevel(1)});
  auto t2 = pb.add(x == Vector2d(1, 3), {PriorityLevel(1), Weight(1)});

  scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});

  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 1)));

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
  auto t1 = pb.add(x == -1., {PriorityLevel(1)});
  auto t2 = pb.add(x == Vector2d(1, 3), {PriorityLevel(1), Weight(1)});

  scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});

  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 1), 1e-10));

  pb.remove(*t1);
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(1, 3), 1e-10));

  pb.add(t1);
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 1), 1e-10));

  pb.add(x == 0., {PriorityLevel(1)});
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 2. / 3), 1e-10));

  pb.remove(*t1);
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(1. / 2, 3. / 2), 1e-10));

  pb.add(t1);
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 2. / 3), 1e-10));

  t1->requirements.weight() = 3;
  pb.remove(*t1);
  pb.add(t1);
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(-0.4, 0), 1e-10));

  t1->requirements.weight() = 100.0;
  pb.remove(*t1);
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(1. / 2, 3. / 2), 1e-10));
}

TEST_CASE("Add/Remove variables from problem")
{
  Space R2(2);
  VariablePtr x = R2.createVariable("x");
  Space R3(3);
  VariablePtr y = R3.createVariable("y");

  LinearizedControlProblem pb;
  auto t1 = pb.add(x == 0., {PriorityLevel(1)});

  scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});

  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 0), 1e-10));

  auto t2 = pb.add(y == 0., {PriorityLevel(1)});
  solver.solve(pb);
  FAST_CHECK_UNARY(x->value().isApprox(Vector2d(0, 0), 1e-10));
  FAST_CHECK_UNARY(y->value().isApprox(Vector3d(0, 0, 0), 1e-10));

  pb.remove(*t1);
  solver.solve(pb);
  FAST_CHECK_UNARY(y->value().isApprox(Vector3d(0, 0, 0), 1e-10));
}

TEST_CASE("Add/Remove constraint with substitutions")
{
  SUBCASE("Simple substitution")
  {
    Space R2(2);
    VariablePtr x = R2.createVariable("x");
    Space R3(3);
    VariablePtr y = R3.createVariable("y");

    VariablePtr y_xy = y->subvariable(R2, "y_xy");

    LinearizedControlProblem pb;

    auto t1 = pb.add(x - y_xy == 0., {PriorityLevel(1)});
    auto t2 = pb.add(y == Vector3d(1, 2, 3), {PriorityLevel(1)});

    scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});

    solver.solve(pb);
    FAST_CHECK_UNARY(x->value().isApprox(Vector2d(1, 2), 1e-10));
    FAST_CHECK_UNARY(y_xy->value().isApprox(Vector2d(1, 2), 1e-10));
    FAST_CHECK_UNARY(y->value().isApprox(Vector3d(1, 2, 3), 1e-10));

    const auto & varInUse = scheme::internal::getComputationData(pb, solver)->variables();
    pb.remove(*t1);
    FAST_CHECK_UNARY_FALSE(pb.variables().contains(*x));
    FAST_CHECK_UNARY(pb.variables().contains(*y));
    FAST_CHECK_UNARY(varInUse.contains(*x)); // Computation data were not updated yet
    FAST_CHECK_UNARY(varInUse.contains(*y));
    solver.solve(pb);
    FAST_CHECK_UNARY(y->value().isApprox(Vector3d(1, 2, 3), 1e-10));
    FAST_CHECK_UNARY_FALSE(varInUse.contains(*x));
    FAST_CHECK_UNARY(varInUse.contains(*y));

    pb.add(t1);
    pb.add(hint::Substitution(pb.constraint(*t1), x));
    solver.solve(pb);
    FAST_CHECK_UNARY(x->value().isApprox(Vector2d(1, 2), 1e-10));
    FAST_CHECK_UNARY(y_xy->value().isApprox(Vector2d(1, 2), 1e-10));
    FAST_CHECK_UNARY(y->value().isApprox(Vector3d(1, 2, 3), 1e-10));
    FAST_CHECK_UNARY_FALSE(varInUse.contains(*x));
    FAST_CHECK_UNARY(varInUse.contains(*y));

    pb.remove(*t1);
    FAST_CHECK_UNARY_FALSE(pb.variables().contains(*x));
    FAST_CHECK_UNARY(pb.variables().contains(*y));
    solver.solve(pb);
    FAST_CHECK_UNARY(y->value().isApprox(Vector3d(1, 2, 3), 1e-10));
    FAST_CHECK_UNARY_FALSE(varInUse.contains(*x));
    FAST_CHECK_UNARY(varInUse.contains(*y));
  }

  SUBCASE("Multiple substitutions")
  {
    Space R(1);
    VariablePtr x = R.createVariable("x");
    VariablePtr y = R.createVariable("y");
    VariablePtr z = R.createVariable("z");
    VariablePtr w = R.createVariable("w");

    LinearizedControlProblem pb;

    auto t1 = pb.add(x + y == 0.);
    auto t2 = pb.add(y + z == 0.);
    auto t3 = pb.add(z + w == 0.);
    auto t4 = pb.add(w == 1.);

    scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});

    auto check = [&](bool bx, bool by, bool bz) {
      const auto & varInUse = scheme::internal::getComputationData(pb, solver)->variables();
      FAST_CHECK_EQ(varInUse.contains(*x), bx);
      FAST_CHECK_EQ(varInUse.contains(*y), by);
      FAST_CHECK_EQ(varInUse.contains(*z), bz);
      FAST_CHECK_UNARY(varInUse.contains(*w));
      FAST_CHECK_EQ(x->value()[0], doctest::Approx(-1));
      FAST_CHECK_EQ(y->value()[0], doctest::Approx(+1));
      FAST_CHECK_EQ(z->value()[0], doctest::Approx(-1));
      FAST_CHECK_EQ(w->value()[0], doctest::Approx(+1));
    };
    solver.solve(pb);
    check(true, true, true);

    // adding all 3 substitutions
    pb.add(hint::Substitution(pb.constraint(*t1), x));
    pb.add(hint::Substitution(pb.constraint(*t2), y));
    pb.add(hint::Substitution(pb.constraint(*t3), z));
    solver.solve(pb);
    check(false, false, false);

    // removing middle one (which is in the middle of the substitution dependency graph)
    pb.removeSubstitutionFor(*pb.constraint(*t2));
    solver.solve(pb);
    check(false, true, false);

    // adding and removing different substitutions between solves
    pb.add(hint::Substitution(pb.constraint(*t2), y));
    pb.removeSubstitutionFor(*pb.constraint(*t1));
    solver.solve(pb);
    check(true, false, false);

    // adding and removing the same substitution
    pb.add(hint::Substitution(pb.constraint(*t1), x));
    pb.removeSubstitutionFor(*pb.constraint(*t1));
    solver.solve(pb);
    check(true, false, false);

    // removing and adding the same substitution
    pb.removeSubstitutionFor(*pb.constraint(*t2));
    pb.add(hint::Substitution(pb.constraint(*t2), y));
    solver.solve(pb);
    check(true, false, false);

    // removing several substitution (including one not present)
    pb.removeSubstitutionFor(*pb.constraint(*t1));
    pb.removeSubstitutionFor(*pb.constraint(*t2));
    pb.removeSubstitutionFor(*pb.constraint(*t3));
    solver.solve(pb);
    check(true, true, true);
  }

  SUBCASE("Complex substitution")
  {
    Space R(1);
    VariablePtr x = R.createVariable("x");
    VariablePtr y = R.createVariable("y");
    VariablePtr z = R.createVariable("z");
    VariablePtr w = R.createVariable("w");

    LinearizedControlProblem pb;

    auto t1 = pb.add(x + y + z + w == 0.);
    auto t2 = pb.add(x - y - z + w == 0.);
    auto t3 = pb.add(x + z + w == 1.);
    auto t4 = pb.add(y + z + w == -1.);

    scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});

    auto check = [&](bool bx, bool bz) {
      const auto & varInUse = scheme::internal::getComputationData(pb, solver)->variables();
      FAST_CHECK_EQ(varInUse.contains(*x), bx);
      FAST_CHECK_UNARY(varInUse.contains(*y));
      FAST_CHECK_EQ(varInUse.contains(*z), bz);
      FAST_CHECK_UNARY(varInUse.contains(*w));
      FAST_CHECK_EQ(x->value()[0], doctest::Approx(+1));
      FAST_CHECK_EQ(y->value()[0], doctest::Approx(-1));
      FAST_CHECK_EQ(z->value()[0], doctest::Approx(+1));
      FAST_CHECK_EQ(w->value()[0], doctest::Approx(-1));
    };
    solver.solve(pb);
    check(true, true);

    pb.add(hint::Substitution({pb.constraint(*t1), pb.constraint(*t2)}, {x, z}));
    solver.solve(pb);
    check(false, false);

    CHECK_THROWS(pb.removeSubstitutionFor(*pb.constraint(*t1)));
    CHECK_NOTHROW(pb.remove(pb.substitutions().substitutions()[0]));
    solver.solve(pb);
    check(true, true);
  }
}

// Skip if more than one type of constraints/objective is added more than once
bool skip(const std::bitset<12> & a)
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
bool skip2(const std::bitset<8> & a)
{
  int eq = a[0] + a[1];
  int ineq = a[2] + a[3];
  int bnd = a[4] + a[5];
  int obj = a[6] + a[7];

  int n1 = (eq > 1) + (ineq > 1) + (bnd > 1) + (obj > 1);
  int n2 = (eq > 0) + (ineq > 0) + (bnd > 0) + (obj > 0);
  return n1 > 1 && n2 > 3;
}

template<std::size_t N>
void buildPb(LinearizedControlProblem & pb,
             const std::array<TaskWithRequirementsPtr, N> & tasks,
             const std::bitset<N> & selection)
{
  for(std::size_t i = 0; i < N; ++i)
  {
    if(selection[i])
      pb.add(tasks[i]);
  }
}

template<std::size_t N>
void checkSolution(const std::array<TaskWithRequirementsPtr, N> & tasks,
                   const std::bitset<N> & added,
                   const LinearizedControlProblem & pb0,
                   const VectorXd & s0,
                   const LinearizedControlProblem & pb,
                   const VectorXd s)
{
  Vector2d eps = Vector2d::Constant(1e-6);
  for(std::size_t j = 0; j < (3 * N) / 4; ++j) // all constraints
  {
    if(added[j])
    {
      auto t = tasks[j]->task;
      auto f = std::static_pointer_cast<tvm::function::abstract::LinearFunction>(t.function());
      pb.variables().value(s);
      f->updateValue();
      Vector2d v = f->value() - tasks[j]->task.template taskDynamics<task_dynamics::None>()->value();
      switch(t.type())
      {
        case constraint::Type::EQUAL:
          FAST_CHECK_UNARY(v.isZero(1e-6));
          break;
        case constraint::Type::GREATER_THAN:
          FAST_CHECK_UNARY((v.array() >= -eps.array()).all());
          break;
        case constraint::Type::LOWER_THAN:
          FAST_CHECK_UNARY((v.array() <= eps.array()).all());
          break;
        case constraint::Type::DOUBLE_SIDED: {
          Vector2d v2 = f->value() - tasks[j]->task.template secondBoundTaskDynamics<task_dynamics::None>()->value();
          FAST_CHECK_UNARY((v.array() >= -eps.array()).all());
          FAST_CHECK_UNARY((v2.array() <= eps.array()).all());
        }
        break;
      }
    }
  }
  for(std::size_t j = (3 * N) / 4; j < N; ++j) // objectives
  {
    if(added[j])
    {
      auto t = tasks[j]->task;
      auto f = std::static_pointer_cast<tvm::function::abstract::LinearFunction>(t.function());
      pb0.variables().value(s0);
      f->updateValue();
      Vector2d obj0 = f->value();
      pb.variables().value(s);
      f->updateValue();
      Vector2d obj = f->value();
      FAST_CHECK_UNARY((obj0 - obj).isZero(5e-5));
    }
  }
}

void test1Change(const std::bitset<12> & selection, bool withSubstitution = false)
{
  if(skip(selection))
    return;

  Space R2(2);
  VariablePtr x = R2.createVariable("x");
  VariablePtr y = R2.createVariable("y");
  VariablePtr z = R2.createVariable("z");

  using task_dynamics::None;
  SolvingRequirements P0 = {PriorityLevel(0)};
  SolvingRequirements P1 = {PriorityLevel(1)};
  // 3 of each: equality, inequality, bound and objective, in that order
  std::array<TaskWithRequirementsPtr, 12> tasks = {
      std::make_shared<TaskWithRequirements>(Task{x + 2 * y == 3., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{3 * x + z == 4., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{y - z == 0., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{x + y >= 0., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{x + y + z >= 1., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{x + z <= 3., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{0. <= x <= 4., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{-2. <= x <= 2., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{x <= 3., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{y == 0., None()}, P1),
      std::make_shared<TaskWithRequirements>(Task{z == 0., None()}, P1),
      std::make_shared<TaskWithRequirements>(Task{x == 0., None()}, P1)};

  IF_USE_LSSOL(scheme::WeightedLeastSquares solverLssol(solver::LSSOLLSSolverOptions{}));
  IF_USE_QLD(scheme::WeightedLeastSquares solverQLD(solver::QLDLSSolverOptions().cholesky(true)));
  IF_USE_QUADPROG(scheme::WeightedLeastSquares solverQuadprog(solver::QuadprogLSSolverOptions().cholesky(true)));
  IF_USE_JRLQP(scheme::WeightedLeastSquares solverJrlqp(solver::JRLQPLSSolverOptions{}));

  int start = withSubstitution ? 1 : 0;
  int stop = withSubstitution ? 11 : 12;
  for(int i = start; i < stop; ++i)
  {
    std::bitset<12> added = selection;
    if(withSubstitution)
    {
      added[0] = true;   // We always want the first constraint, for substitution
      added[11] = false; // Since x is substituted, we make sure the objective is full rank (due to current limitations
                         // of QLD and quadprog)
    }
    LinearizedControlProblem pb;
    buildPb(pb, tasks, added);
    if(withSubstitution)
      pb.add(hint::Substitution(pb.constraint(*tasks[0]), x));

    // We solve pb for the current list of tasks
    IF_USE_LSSOL(solverLssol.solve(pb));
    IF_USE_QLD(solverQLD.solve(pb));
    IF_USE_QUADPROG(solverQuadprog.solve(pb));
    IF_USE_JRLQP(solverJrlqp.solve(pb));

    if(added[i])
    {
      pb.remove(*tasks[i]);
      added[i] = false;
    }
    else
    {
      pb.add(tasks[i]);
      added[i] = true;
    }

    // Create a ground truth problem
    LinearizedControlProblem pbGroundTruth;
    buildPb(pbGroundTruth, tasks, added);

    // Now we solve both problem. Only pb is performing an update
    tvm::utils::set_is_malloc_allowed(false);
#if defined(TVM_USE_LSSOL)
    FAST_CHECK_UNARY(solverLssol.solve(pbGroundTruth));
#elif defined(TVM_USE_QLD)
    FAST_CHECK_UNARY(solverQLD.solve(pbGroundTruth));
#elif defined(TVM_USE_QUADPROG)
    FAST_CHECK_UNARY(solverQuadprog.solve(pbGroundTruth));
#elif defined(TVM_USE_JRLQP)
    FAST_CHECK_UNARY(solverJrlqp.solve(pbGroundTruth));
#endif
    VectorXd s0 = pbGroundTruth.variables().value();

#ifdef TVM_USE_LSSOL
    FAST_CHECK_UNARY(solverLssol.solve(pb));
    checkSolution(tasks, added, pbGroundTruth, s0, pb, pb.variables().value());
#endif
#ifdef TVM_USE_QLD
    FAST_CHECK_UNARY(solverQLD.solve(pb));
    checkSolution(tasks, added, pbGroundTruth, s0, pb, pb.variables().value());
#endif
#ifdef TVM_USE_QUADPROG
    FAST_CHECK_UNARY(solverQuadprog.solve(pb));
    checkSolution(tasks, added, pbGroundTruth, s0, pb, pb.variables().value());
#endif
#ifdef TVM_USE_JRLQP
    FAST_CHECK_UNARY(solverJrlqp.solve(pb));
    checkSolution(tasks, added, pbGroundTruth, s0, pb, pb.variables().value());
#endif
    tvm::utils::set_is_malloc_allowed(true);
  }
}

void test3Change(const std::bitset<8> & selection)
{
  if(skip2(selection))
    return;

  Space R2(2);
  VariablePtr x = R2.createVariable("x");
  VariablePtr y = R2.createVariable("y");
  VariablePtr z = R2.createVariable("z");

  using task_dynamics::None;
  SolvingRequirements P0 = {PriorityLevel(0)};
  SolvingRequirements P1 = {PriorityLevel(1)};
  // 2 of each: equality, inequality, bound and objective, in that order
  std::array<TaskWithRequirementsPtr, 8> tasks = {
      std::make_shared<TaskWithRequirements>(Task{x + 2 * y == 3., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{3 * x + z == 4., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{x + y >= 0., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{x + z <= 3., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{0. <= x <= 4., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{-2. <= x <= 2., None()}, P0),
      std::make_shared<TaskWithRequirements>(Task{y == 0., None()}, P1),
      std::make_shared<TaskWithRequirements>(Task{z == 0., None()}, P1)};

  IF_USE_LSSOL(scheme::WeightedLeastSquares solverLssol(solver::LSSOLLSSolverOptions{}));
  IF_USE_QLD(scheme::WeightedLeastSquares solverQLD(solver::QLDLSSolverOptions().cholesky(true)));
  IF_USE_QUADPROG(scheme::WeightedLeastSquares solverQuadprog(solver::QuadprogLSSolverOptions().cholesky(true)));
  IF_USE_JRLQP(scheme::WeightedLeastSquares solverJrlqp(solver::JRLQPLSSolverOptions{}));

  for(int i = 0; i < 8; ++i)
  {
    for(int j = 0; j < 8; ++j)
    {
      for(int k = 0; k < 8; k += 2)
      {
        std::bitset<8> added = selection;
        LinearizedControlProblem pb;
        buildPb(pb, tasks, added);

        // We solve pb for the current list of tasks
        IF_USE_LSSOL(solverLssol.solve(pb));
        IF_USE_QLD(solverQLD.solve(pb));
        IF_USE_QUADPROG(solverQuadprog.solve(pb));
        IF_USE_JRLQP(solverJrlqp.solve(pb));

        if(added[i])
        {
          pb.remove(*tasks[i]);
          added[i] = false;
        }
        else
        {
          pb.add(tasks[i]);
          added[i] = true;
        }

        if(added[j])
        {
          pb.remove(*tasks[j]);
          added[j] = false;
        }
        else
        {
          pb.add(tasks[j]);
          added[j] = true;
        }

        if(added[k])
        {
          pb.remove(*tasks[k]);
          added[k] = false;
        }
        else
        {
          pb.add(tasks[k]);
          added[k] = true;
        }

        // Create a ground truth problem
        LinearizedControlProblem pbGroundTruth;
        buildPb(pbGroundTruth, tasks, added);

        // Now we solve both problem. Only pb is performing an update
        tvm::utils::set_is_malloc_allowed(false);
#if defined(TVM_USE_LSSOL)
        FAST_CHECK_UNARY(solverLssol.solve(pbGroundTruth));
#elif defined(TVM_USE_QLD)
        FAST_CHECK_UNARY(solverQLD.solve(pbGroundTruth));
#elif defined(TVM_USE_QUADPROG)
        FAST_CHECK_UNARY(solverQuadprog.solve(pbGroundTruth));
#elif defined(TVM_USE_JRLQP)
        FAST_CHECK_UNARY(solverJrlqp.solve(pbGroundTruth));
#endif
        VectorXd s0 = pbGroundTruth.variables().value();

#ifdef TVM_USE_LSSOL
        FAST_CHECK_UNARY(solverLssol.solve(pb));
        checkSolution(tasks, added, pbGroundTruth, s0, pb, pb.variables().value());
#endif
#ifdef TVM_USE_QLD
        FAST_CHECK_UNARY(solverQLD.solve(pb));
        checkSolution(tasks, added, pbGroundTruth, s0, pb, pb.variables().value());
#endif
#ifdef TVM_USE_QUADPROG
        FAST_CHECK_UNARY(solverQuadprog.solve(pb));
        checkSolution(tasks, added, pbGroundTruth, s0, pb, pb.variables().value());
#endif
#ifdef TVM_USE_JRLQP
        FAST_CHECK_UNARY(solverJrlqp.solve(pb));
        checkSolution(tasks, added, pbGroundTruth, s0, pb, pb.variables().value());
#endif
        tvm::utils::set_is_malloc_allowed(true);
      }
    }
  }
}

#ifdef TVM_THOROUGH_TESTING
TEST_CASE("Systematic add/remove of one task")
{
  // This test is creating a large number of problems, leading to a large log.
  // So we disable the logs here.
  tvm::graph::internal::Logger::logger().disable();

  std::bitset<12> added(0);
  // all combinations of true/false for added
  for(uint16_t i = 1; i < (1 << 12); ++i)
  {
    added = i;
    test1Change(added);
  }
}
#else
TEST_CASE("Some add/remove of one task")
{
  // This test is creating a large number of problems, leading to a large log.
  // So we disable the logs here.
  tvm::graph::internal::Logger::logger().disable();

  std::vector<std::bitset<12>> added = {
      std::bitset<12>("110000000000"), std::bitset<12>("110100000000"), std::bitset<12>("110000100000"),
      std::bitset<12>("110000000100"), std::bitset<12>("000110000000"), std::bitset<12>("100110000000"),
      std::bitset<12>("000110100000"), std::bitset<12>("000110000100"), std::bitset<12>("000000110000"),
      std::bitset<12>("100000110000"), std::bitset<12>("000100110000"), std::bitset<12>("000000110100"),
      std::bitset<12>("000000000110"), std::bitset<12>("100000000110"), std::bitset<12>("000100000110"),
      std::bitset<12>("000000100110")};
  for(size_t i = 0; i < added.size(); ++i)
  {
    test1Change(added[i]);
  }
}
#endif

#ifdef TVM_THOROUGH_TESTING
TEST_CASE("Systematic add/remove of one task with substitution")
{
  // This test is creating a large number of problems, leading to a large log.
  // So we disable the logs here.
  tvm::graph::internal::Logger::logger().disable();

  std::bitset<12> added(1);
  // all combinations of true/false for added[1..10]
  for(int i = 1; i < (1 << 10); ++i)
  {
    added = 1 + 2 * i + (1 << 11);
    test1Change(added, true);
  }
}
#else
TEST_CASE("Some add/remove of one task with substitution")
{
  // This test is creating a large number of problems, leading to a large log.
  // So we disable the logs here.
  tvm::graph::internal::Logger::logger().disable();

  std::vector<std::bitset<12>> added = {
      std::bitset<12>("110000000000"), std::bitset<12>("110000100000"), std::bitset<12>("110000100000"),
      std::bitset<12>("110000000100"), std::bitset<12>("100110000000"), std::bitset<12>("100110100000"),
      std::bitset<12>("100110000100"), std::bitset<12>("100000110000"), std::bitset<12>("100100110000"),
      std::bitset<12>("100000110100"), std::bitset<12>("100000000110"), std::bitset<12>("100100000110"),
      std::bitset<12>("100000100110")};
  for(size_t i = 0; i < added.size(); ++i)
  {
    test1Change(added[i], true);
  }
}
#endif

#ifdef TVM_THOROUGH_TESTING
TEST_CASE("Systematic add/remove of three tasks")
{
  // This test is creating a large number of problems, leading to a large log.
  // So we disable the logs here.
  tvm::graph::internal::Logger::logger().disable();

  std::bitset<8> added(0);
  // all combinations of true/false for added
  for(int i = 1; i < (1 << 8); ++i)
  {
    added = i;
    test3Change(added);
  }
}
#else
TEST_CASE("Some add/remove of three tasks")
{
  // This test is creating a large number of problems, leading to a large log.
  // So we disable the logs here.
  tvm::graph::internal::Logger::logger().disable();

  std::vector<std::bitset<8>> added = {std::bitset<8>("11000000"), std::bitset<8>("00110000"),
                                       std::bitset<8>("00001100"), std::bitset<8>("00000011")};
  for(size_t i = 0; i < added.size(); ++i)
  {
    test3Change(added[i]);
  }
}
#endif
