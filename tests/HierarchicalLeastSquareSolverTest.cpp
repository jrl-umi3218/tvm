/* Copyright 2022 CNRS-AIST JRL and CNRS-UM LIRMM */
//#if TVM_WITH_LEXLS
#include <tvm/solver/LexLSHierarchicalLeastSquareSolver.h>
//#endif

#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/constraint/BasicLinearConstraint.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/scheme/HierarchicalLeastSquares.h>
#include <tvm/task_dynamics/None.h>

#include <Eigen/SVD>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/EigenDoctest.h"

using namespace tvm;
using namespace tvm::constraint;
using namespace tvm::requirements;
using namespace tvm::solver;
using namespace Eigen;

MatrixXd pinv(const MatrixConstRef & M, double eps = 1e-10)
{
  auto svd = M.jacobiSvd(ComputeThinU | ComputeThinV);
  svd.setThreshold(eps);
  int r = svd.rank();
  return svd.matrixV().leftCols(r) * svd.singularValues().head(r).cwiseInverse().asDiagonal()
         * svd.matrixU().leftCols(r).transpose();
}

TEST_CASE("LexLSHierarchicalLeastSquareSolver")
{
  VariablePtr x = Space(6).createVariable("x");
  MatrixXd A0(3, 6), A1(3, 6), A2(3, 6);
  VectorXd b0(3), b1(3), b2(3);
  // rank(A0)=2
  A0 = MatrixXd::Random(3, 2) * MatrixXd::Random(2, 6);
  // rank(A1) = 3 but rank(A1 projected on the nullspace of A0) = 2
  A1 << MatrixXd::Random(1, 3) * A0, MatrixXd::Random(2, 6);
  A1 = MatrixXd::Random(3, 3) * A1;
  A2.setRandom();
  b0.setRandom();
  b1.setRandom();
  b2.setRandom();
  auto c0 = std::make_shared<BasicLinearConstraint>(A0, x, b0, constraint::Type::EQUAL);
  auto c1a = std::make_shared<BasicLinearConstraint>(A1.row(0), x, b1.head(1), constraint::Type::EQUAL);
  auto c1b = std::make_shared<BasicLinearConstraint>(A1.row(1), x, b1.segment(1, 1), constraint::Type::EQUAL);
  auto c1c = std::make_shared<BasicLinearConstraint>(A1.row(2), x, b1.tail(1), constraint::Type::EQUAL);
  auto c2 = std::make_shared<BasicLinearConstraint>(A2, x, b2, constraint::Type::EQUAL);

  auto r0 = std::make_shared<SolvingRequirementsWithCallbacks>(PriorityLevel(0));
  auto r1 = std::make_shared<SolvingRequirementsWithCallbacks>(PriorityLevel(1));
  auto r2 = std::make_shared<SolvingRequirementsWithCallbacks>(PriorityLevel(2));

  LexLSHierarchicalLeastSquareSolver solver(LexLSHLSSolverOptions{});
  VariableVector vars(x);
  solver.startBuild(vars, {3, 3, 3}, {0, 0, 0}, false);
  solver.addConstraint(c0, r0);
  solver.addConstraint(c1a, r1);
  solver.addConstraint(c1b, r1);
  solver.addConstraint(c1c, r1);
  solver.addConstraint(c2, r2);
  solver.finalizeBuild();

  solver.solve();

  // Compute the solution by hand
  MatrixXd P0 = MatrixXd::Identity(6, 6) - pinv(A0) * A0;
  MatrixXd A01(6, 6);
  A01 << A0, A1;
  MatrixXd P1 = MatrixXd::Identity(6, 6) - pinv(A01) * A01;
  VectorXd dx0 = pinv(A0) * b0;
  VectorXd dx1 = pinv(A1 * P0) * (b1 - A1 * dx0);
  VectorXd dx2 = pinv(A2 * P1) * (b2 - A2 * (dx0 + dx1));
  VectorXd x0 = dx0 + dx1 + dx2;

  FAST_CHECK_EQ(solver.result(), Approx(x0).epsilon(1e-10));
}

TEST_CASE("LexLSHierarchicalLeastSquareSolver min norm")
{
  VariablePtr x = Space(6).createVariable("x");
  MatrixXd A0(3, 6), A1(3, 6);
  VectorXd b0(3), b1(3);
  // rank(A0)=2
  A0 = MatrixXd::Random(3, 2) * MatrixXd::Random(2, 6);
  // rank(A1) = 3 but rank(A1 projected on the nullspace of A0) = 2
  A1 << MatrixXd::Random(1, 3) * A0, MatrixXd::Random(2, 6);
  A1 = MatrixXd::Random(3, 3) * A1;
  b0.setRandom();
  b1.setRandom();
  auto c0 = std::make_shared<BasicLinearConstraint>(A0, x, b0, constraint::Type::EQUAL);
  auto c1 = std::make_shared<BasicLinearConstraint>(A1, x, b1, constraint::Type::EQUAL);

  auto r0 = std::make_shared<SolvingRequirementsWithCallbacks>(PriorityLevel(0));
  auto r1 = std::make_shared<SolvingRequirementsWithCallbacks>(PriorityLevel(1));

  LexLSHierarchicalLeastSquareSolver solver(LexLSHLSSolverOptions{});
  VariableVector vars(x);
  solver.startBuild(vars, {3, 3, 6}, {0, 0, 0}, false);
  solver.addConstraint(c0, r0);
  solver.addConstraint(c1, r1);
  solver.setMinimumNorm(); // Equivalent to A2 = I and b2 = 0
  solver.finalizeBuild();

  solver.solve();

  // Compute the solution by hand
  MatrixXd P0 = MatrixXd::Identity(6, 6) - pinv(A0) * A0;
  MatrixXd A01(6, 6);
  A01 << A0, A1;
  MatrixXd P1 = MatrixXd::Identity(6, 6) - pinv(A01) * A01;
  VectorXd dx0 = pinv(A0) * b0;
  VectorXd dx1 = pinv(A1 * P0) * (b1 - A1 * dx0);
  VectorXd dx2 = pinv(P1) * (-(dx0 + dx1));
  VectorXd x0 = dx0 + dx1 + dx2;

  FAST_CHECK_EQ(solver.result(), Approx(x0).epsilon(1e-10));
}

TEST_CASE("HierarchicalLeastSquares")
{
  SUBCASE("Equality only")
  {
    VariablePtr x = Space(6).createVariable("x");
    MatrixXd A0(3, 6), A1(3, 6), A2(3, 6);
    VectorXd b0(3), b1(3), b2(3);
    // rank(A0)=2
    A0 = MatrixXd::Random(3, 2) * MatrixXd::Random(2, 6);
    // rank(A1) = 3 but rank(A1 projected on the nullspace of A0) = 2
    A1 << MatrixXd::Random(1, 3) * A0, MatrixXd::Random(2, 6);
    A1 = MatrixXd::Random(3, 3) * A1;
    A2.setRandom();
    b0.setRandom();
    b1.setRandom();
    b2.setRandom();

    LinearizedControlProblem pb;
    pb.add(A0 * x - b0 == 0., {PriorityLevel(0)});
    pb.add(A1 * x - b1 == 0., {PriorityLevel(1)});
    pb.add(A2 * x - b2 == 0., {PriorityLevel(2)});

    scheme::HierarchicalLeastSquares solver(LexLSHLSSolverOptions{});

    solver.solve(pb);

    // Compute the solution by hand
    MatrixXd P0 = MatrixXd::Identity(6, 6) - pinv(A0) * A0;
    MatrixXd A01(6, 6);
    A01 << A0, A1;
    MatrixXd P1 = MatrixXd::Identity(6, 6) - pinv(A01) * A01;
    VectorXd dx0 = pinv(A0) * b0;
    VectorXd dx1 = pinv(A1 * P0) * (b1 - A1 * dx0);
    VectorXd dx2 = pinv(A2 * P1) * (b2 - A2 * (dx0 + dx1));
    VectorXd x0 = dx0 + dx1 + dx2;

    FAST_CHECK_EQ(x->value(), Approx(x0).epsilon(1e-10));
  }

  SUBCASE("Inequality only")
  {
    VariablePtr x = Space(1).createVariable("x");
    VariablePtr y = Space(1).createVariable("y");

    LinearizedControlProblem pb;
    pb.add(-1. <= x <= 1., {PriorityLevel(0)});
    pb.add(2. <= x <= 3., {PriorityLevel(1)});
    pb.add(-1. <= y <= 1., {PriorityLevel(2)});
    pb.add(x + y <= 0., {PriorityLevel(4)});

    scheme::HierarchicalLeastSquares solver(LexLSHLSSolverOptions().feasibleFirstLevel(true));

    solver.solve(pb);
    FAST_CHECK_EQ(x->value()[0], doctest::Approx(1).epsilon(1e-10));
    FAST_CHECK_EQ(y->value()[0], doctest::Approx(-1).epsilon(1e-10));
  }
}
