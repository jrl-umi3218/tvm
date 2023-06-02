/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Variable.h>
#include <tvm/function/BasicLinearFunction.h>
#include <tvm/utils/memoryChecks.h>

#include <Eigen/Core>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace Eigen;
using namespace tvm;

TEST_CASE("Test subvariable jacobians")
{
  VariablePtr x = Space(7).createVariable("x");
  VariablePtr y = Space(8).createVariable("y");
  VariablePtr z = Space(6).createVariable("z");
  VariablePtr w = Space(4).createVariable("w");

  VariablePtr x1 = x->subvariable(3, "x1", 1);
  VariablePtr y1 = y->subvariable(6, "y1", 1);
  VariablePtr y2 = y->subvariable(3, "y2", 2);

  MatrixXd A = MatrixXd::Random(3, 7);
  MatrixXd B = MatrixXd::Random(3, 6);
  MatrixXd C = MatrixXd::Random(3, 6);

  function::BasicLinearFunction f(A * x + B * y1 + C * z);

  tvm::utils::set_is_malloc_allowed(false);
  FAST_CHECK_UNARY(f.jacobian(*x).isApprox(A));
  FAST_CHECK_UNARY(f.jacobian(*x1).isApprox(A.middleCols(1, 3)));
  FAST_CHECK_UNARY(f.jacobian(*y1).isApprox(B));
  FAST_CHECK_UNARY(f.jacobian(*y2).isApprox(B.middleCols(1, 3)));
  CHECK_THROWS(f.jacobian(*y));
  CHECK_THROWS(f.jacobian(*w));
  tvm::utils::set_is_malloc_allowed(true);
}

TEST_CASE("Test subvariable JDot")
{
  VariablePtr x = Space(7).createVariable("x");
  VariablePtr y = Space(8).createVariable("y");
  VariablePtr z = Space(6).createVariable("z");
  VariablePtr w = Space(4).createVariable("w");

  VariablePtr x1 = x->subvariable(3, "x1", 1);
  VariablePtr y1 = y->subvariable(6, "y1", 1);
  VariablePtr y2 = y->subvariable(3, "y2", 2);

  MatrixXd A = MatrixXd::Random(3, 7);
  MatrixXd B = MatrixXd::Random(3, 6);
  MatrixXd C = MatrixXd::Random(3, 6);

  function::BasicLinearFunction f(A * x + B * y1 + C * z);

  tvm::utils::set_is_malloc_allowed(false);
  FAST_CHECK_EQ(f.JDot(*x).cols(), 7);
  FAST_CHECK_EQ(f.JDot(*x1).cols(), 3);
  FAST_CHECK_EQ(f.JDot(*y1).cols(), 6);
  FAST_CHECK_EQ(f.JDot(*y2).cols(), 3);
  CHECK_THROWS(f.JDot(*y));
  CHECK_THROWS(f.JDot(*w));
  tvm::utils::set_is_malloc_allowed(true);
}
