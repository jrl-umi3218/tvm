//#if TVM_WITH_LEXLS
#include <tvm/solver/LexLSHierarchicalLeastSquareSolver.h>
//#endif

#include <tvm/Variable.h>
#include <tvm/constraint/BasicLinearConstraint.h>

#include <eigen/SVD>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm;
using namespace tvm::constraint;
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

  LexLSHierarchicalLeastSquareSolver solver(LexLSHLSSolverOptions{});
  VariableVector vars(x);
  solver.startBuild(vars, {3, 3, 3}, {0, 0, 0}, false);
  solver.addConstraint(0, c0);
  solver.addConstraint(1, c1a);
  solver.addConstraint(1, c1b);
  solver.addConstraint(1, c1c);
  solver.addConstraint(2, c2);
  solver.finalizeBuild();

  solver.solve();

  MatrixXd P0 = MatrixXd::Identity(6, 6) - pinv(A0) * A0;
  MatrixXd A01(6, 6);
  A01 << A0, A1;
  MatrixXd P1 = MatrixXd::Identity(6, 6) - pinv(A01) * A01;
  VectorXd dx0 = pinv(A0) * b0;
  VectorXd dx1 = pinv(A1 * P0) * (b1 - A1 * dx0);
  VectorXd dx2 = pinv(A2 * P1) * (b2 - A2 * (dx0 + dx1));
  VectorXd x0 = dx0 + dx1 + dx2;

  //std::cout << solver.result().transpose() << std::endl;
  //std::cout << x0.transpose() << std::endl;

  FAST_CHECK_UNARY(x0.isApprox(solver.result(), 1e-10));
}