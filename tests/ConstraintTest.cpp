#include "LinearConstraint.h"
#include "Variable.h"
#include "exceptions.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"


using namespace Eigen;
using namespace tvm;

TEST_CASE("Test constraint")
{
  MatrixXd A1 = MatrixXd::Random(4, 5);
  MatrixXd A2 = MatrixXd::Random(4, 2);
  VectorXd b = VectorXd::Random(4);

  VariablePtr x1 = Space(5).createVariable("x1");
  VariablePtr x2 = Space(2).createVariable("x2");

  x1->value(VectorXd::Random(5));
  x2->value(VectorXd::Random(2));

  //A1x >= 0
  BasicLinearConstraint C1(A1, x1, ConstraintType::GREATER_THAN);
  C1.updateValue();

  FAST_CHECK_UNARY(C1.value().isApprox(A1*x1->value()));

  // [A1 A2] [x1' x2']' <= b
  BasicLinearConstraint C2({ A1,A2 }, { x1,x2 }, b, ConstraintType::LOWER_THAN);
  C2.updateValue();
  BasicLinearConstraint C3({ A2,A1 }, { x2,x1 }, b, ConstraintType::LOWER_THAN);
  C3.updateValue();

  FAST_CHECK_UNARY(C2.value().isApprox(A1*x1->value() + A2*x2->value()));
  FAST_CHECK_UNARY(C2.value().isApprox(C3.value()));
  FAST_CHECK_UNARY(C2.u().isApprox(b));
}
