#include <tvm/Variable.h>
#include <tvm/constraint/BasicLinearConstraint.h>
#include <tvm/exception/exceptions.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"


TEST_CASE("Test constraint")
{
  Eigen::MatrixXd A1 = Eigen::MatrixXd::Random(4, 5);
  Eigen::MatrixXd A2 = Eigen::MatrixXd::Random(4, 2);
  Eigen::VectorXd b = Eigen::VectorXd::Random(4);

  tvm::VariablePtr x1 = tvm::Space(5).createVariable("x1");
  tvm::VariablePtr x2 = tvm::Space(2).createVariable("x2");

  x1->value(Eigen::VectorXd::Random(5));
  x2->value(Eigen::VectorXd::Random(2));

  //A1x >= 0
  tvm::constraint::BasicLinearConstraint C1(A1, x1, tvm::constraint::Type::GREATER_THAN);
  C1.updateValue();

  FAST_CHECK_UNARY(C1.value().isApprox(A1*x1->value()));

  // [A1 A2] [x1' x2']' <= b
  tvm::constraint::BasicLinearConstraint C2({ A1,A2 }, { x1,x2 }, b, tvm::constraint::Type::LOWER_THAN);
  C2.updateValue();
  tvm::constraint::BasicLinearConstraint C3({ A2,A1 }, { x2,x1 }, b, tvm::constraint::Type::LOWER_THAN);
  C3.updateValue();

  FAST_CHECK_UNARY(C2.value().isApprox(A1*x1->value() + A2*x2->value()));
  FAST_CHECK_UNARY(C2.value().isApprox(C3.value()));
  FAST_CHECK_UNARY(C2.u().isApprox(b));
}
