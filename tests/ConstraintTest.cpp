#include "LinearConstraint.h"
#include "Variable.h"

#include <iostream>

using namespace Eigen;
using namespace tvm;
void linearConstraintTest()
{
  MatrixXd A1 = MatrixXd::Random(4, 5);
  MatrixXd A2 = MatrixXd::Random(4, 2);
  MatrixXd A3 = MatrixXd::Random(4, 4);
  VectorXd b = VectorXd::Random(4);

  std::shared_ptr<Variable> x1 = Space(5).createVariable("x1");
  std::shared_ptr<Variable> x2 = Space(2).createVariable("x2");
  std::shared_ptr<Variable> x3 = Space(4).createVariable("x3");

  x1->setValue(VectorXd::Random(5));
  x2->setValue(VectorXd::Random(2));
  x3->setValue(VectorXd::Random(4));

  //A1x >= 0
  BasicLinearConstraint C1(A1, x1, ConstraintType::GREATER_THAN);

  // [A1 A2] [x1' x2']' <= b
  BasicLinearConstraint C2({ A1,A2 }, { x1,x2 }, b, ConstraintType::LOWER_THAN);
  BasicLinearConstraint C3({ A2,A1 }, { x2,x1 }, b, ConstraintType::LOWER_THAN);

  std::cout << C2.value().isApprox(C3.value()) << std::endl;
  std::cout << C2.u() << std::endl;
  std::cout << C2.l() << std::endl;
}