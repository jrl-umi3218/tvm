#include "LinearConstraint.h"
#include "Variable.h"
#include "exceptions.h"

// boost
#define BOOST_TEST_MODULE ConstraintTest
#include <boost/test/unit_test.hpp>

using namespace Eigen;
using namespace tvm;

BOOST_AUTO_TEST_CASE(ConstraintTest)
{
  MatrixXd A1 = MatrixXd::Random(4, 5);
  MatrixXd A2 = MatrixXd::Random(4, 2);
  VectorXd b = VectorXd::Random(4);

  std::shared_ptr<Variable> x1 = Space(5).createVariable("x1");
  std::shared_ptr<Variable> x2 = Space(2).createVariable("x2");

  x1->value(VectorXd::Random(5));
  x2->value(VectorXd::Random(2));

  //A1x >= 0
  BasicLinearConstraint C1(A1, x1, ConstraintType::GREATER_THAN);
  C1.updateValue();

  BOOST_CHECK(C1.value().isApprox(A1*x1->value()));
  BOOST_CHECK_THROW(C1.u(), tvm::UnusedOutput);
  BOOST_CHECK_THROW(C1.l(), tvm::UnusedOutput);

  // [A1 A2] [x1' x2']' <= b
  BasicLinearConstraint C2({ A1,A2 }, { x1,x2 }, b, ConstraintType::LOWER_THAN);
  C2.updateValue();
  BasicLinearConstraint C3({ A2,A1 }, { x2,x1 }, b, ConstraintType::LOWER_THAN);
  C3.updateValue();

  BOOST_CHECK(C2.value().isApprox(A1*x1->value() + A2*x2->value()));
  BOOST_CHECK(C2.value().isApprox(C3.value()));
  BOOST_CHECK(C2.u().isApprox(b));
  BOOST_CHECK_THROW(C2.l(), tvm::UnusedOutput);
}
