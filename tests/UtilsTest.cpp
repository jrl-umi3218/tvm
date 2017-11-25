#include "SolverTestFunctions.h"
#include <tvm/Task.h>
#include <tvm/Variable.h>
#include <tvm/function/BasicLinearFunction.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/task_dynamics/Proportional.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/utils/ProtoTask.h>
#include <tvm/utils/UpdateLessFunction.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace Eigen;
using namespace tvm;
using namespace tvm::utils;


template<typename T>
void checkSimpleBoundOnFunction(const T& bound)
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  auto f = std::make_shared<function::IdentityFunction>(x);
  task_dynamics::P td(2);

  {
    Task t(f == bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::EQUAL);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound == f, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::EQUAL);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(f >= bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::GREATER_THAN);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound <= f, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::GREATER_THAN);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(f <= bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::LOWER_THAN);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound >= f, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::LOWER_THAN);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }
}


template<typename L, typename U>
void checkDoubleBoundsOnFunction(const L& l, const U& u)
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  auto f = std::make_shared<function::IdentityFunction>(x);
  task_dynamics::P td(2);

  {
    Task t(l <= f <= u, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::DOUBLE_SIDED);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY(t.secondBoundTaskDynamics());
  }

  {
    Task t(l >= f >= u, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::DOUBLE_SIDED);
    FAST_CHECK_EQ(t.function(), f);
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY(t.secondBoundTaskDynamics());
  }
}

template<typename T>
void checkSimpleBoundOnVariable(const T& bound)
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  task_dynamics::P td(2);

  {
    Task t(x == bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::EQUAL);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound == x, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::EQUAL);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(x >= bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::GREATER_THAN);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound <= x, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::GREATER_THAN);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(x <= bound, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::LOWER_THAN);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }

  {
    Task t(bound >= x, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::LOWER_THAN);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY_FALSE(t.secondBoundTaskDynamics());
  }
}


template<typename L, typename U>
void checkDoubleBoundsOnVariable(const L& l, const U& u)
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  task_dynamics::P td(2);

  {
    Task t(l <= x <= u, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::DOUBLE_SIDED);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY(t.secondBoundTaskDynamics());
  }

  {
    Task t(l >= x >= u, td);
    FAST_CHECK_EQ(t.type(), constraint::Type::DOUBLE_SIDED);
    FAST_CHECK_UNARY(t.function()->jacobian(*x).properties().isIdentity());
    CHECK_NOTHROW(t.taskDynamics<task_dynamics::P>());
    CHECK_THROWS(t.taskDynamics<task_dynamics::PD>());
    FAST_CHECK_UNARY(t.secondBoundTaskDynamics());
  }
}

TEST_CASE("Test Proto task")
{
  // scalar-defined bounds
  checkSimpleBoundOnFunction(0.);
  checkDoubleBoundsOnFunction(-1., 3.);
  checkSimpleBoundOnVariable(0.);
  checkDoubleBoundsOnVariable(-1., 3.);

  // vector-defined bounds
  VectorXd l = -Vector2d::Ones();
  VectorXd u = Vector2d::Ones();
  checkSimpleBoundOnFunction(l);
  checkDoubleBoundsOnFunction(l, u);
  checkSimpleBoundOnVariable(l);
  checkDoubleBoundsOnVariable(l, u);

  // mix
  checkDoubleBoundsOnFunction(-1., u);
  checkDoubleBoundsOnFunction(l, 3.);
  checkDoubleBoundsOnVariable(-1., u);
  checkDoubleBoundsOnVariable(l, 3.);

  // expression
  Matrix2d A = Matrix2d::Identity();
  checkSimpleBoundOnFunction(-l);
  checkSimpleBoundOnFunction(A*l - u);
  checkSimpleBoundOnVariable(-l);
  checkSimpleBoundOnVariable(A*l - u);
}

TEST_CASE("Test UpdatelessFunction")
{
  //value
  VariablePtr x = Space(2).createVariable("x");
  VariablePtr y = Space(4).createVariable("y");
  VariablePtr z = Space(3).createVariable("z");
  MatrixXd Ax = MatrixXd::Random(3, 2);
  MatrixXd Ay = MatrixXd::Random(3, 4);
  MatrixXd Az = MatrixXd::Random(3, 3);
  auto f = std::shared_ptr<function::BasicLinearFunction>( 
    new function::BasicLinearFunction( { Ax, Ay, Az }, {x, y, z} ));

  UpdatelessFunction uf(f);

  VectorXd xr = VectorXd::Random(2);
  VectorXd yr = VectorXd::Random(4);
  VectorXd zr = VectorXd::Random(3);

  VectorXd v = uf.value(xr, yr, zr);
  FAST_CHECK_UNARY(v.isApprox(Ax * xr + Ay * yr + Az * zr));
  FAST_CHECK_UNARY(x->value().isApprox(xr));
  FAST_CHECK_UNARY(y->value().isApprox(yr));
  FAST_CHECK_UNARY(z->value().isApprox(zr));

  VectorXd xm(2); xm << 1, 2;
  VectorXd ym(4); ym << 3, 4, 5, 6;
  VectorXd zm(3); zm << 7, 8, 9;

  using l = std::initializer_list<double>;
  v = uf.value(l{ 1,2 }, l{ 3,4,5,6 }, l{ 7., 8., 9. });
  FAST_CHECK_UNARY(v.isApprox(Ax * xm + Ay * ym + Az * zm));

  v = uf.value(l{ 1,2 }, yr, l{ 7., 8., 9. });
  FAST_CHECK_UNARY(v.isApprox(Ax * xm + Ay * yr + Az * zm));

  v = uf.value(*x, xr);
  FAST_CHECK_UNARY(v.isApprox(Ax * xr + Ay * yr + Az * zm));

  v = uf.value(*z, l{ 1,2,3 }, *y, l{ 3,4,5,6 }, *z, l{ 7,8,9 }, *x, xr);
  FAST_CHECK_UNARY(v.isApprox(Ax * xr + Ay * ym + Az * zm));

  //errors
  // not enough args
  CHECK_THROWS(uf.value(xr, yr));
  CHECK_THROWS(uf.value(l{ 1,2 }, l{3,4,5,6}));
  //uf.value(*x); //does not compile (and that's normal)
  // too many args
  CHECK_THROWS(uf.value(xr, yr, zr, xm, ym, zm));
  CHECK_THROWS(uf.value(l{ 1,2 }, l{ 3,4,5,6 }, zr, l{ 7,8,9 }));
  //uf.value(*x, xr, yr); //does not compile (and that's normal)
  // wrong size
  CHECK_THROWS(uf.value(xr, zr, yr));
  CHECK_THROWS(uf.value(l{ 1,2,3 }, l{ 3,4,5,6 }, l{ 7., 8., 9. }));
  CHECK_THROWS(uf.value(*x, zr));


  //jacobian
  auto s1 = std::make_shared<SphereFunction>(x, Vector2d(0, 0), 1);
  auto s2 = std::make_shared<SphereFunction>(z, Vector3d(0, 0, 1), 1);
  auto df = std::make_shared<Difference>(s1, s2);
  UpdatelessFunction udf(df);

  MatrixXd J = udf.jacobian(*x, xr, zr);
  FAST_CHECK_UNARY(J.isApprox(2 * xr.transpose()));
  J = udf.jacobian(*z, xr, zr);
  FAST_CHECK_UNARY(J.isApprox(- 2 * zr.transpose()));
  J = udf.jacobian(*x, l{ 1,2 }, l{ 7,8,9 });
  FAST_CHECK_UNARY(J.isApprox(2 * xm.transpose()));
  J = udf.jacobian(*x, *x, l{ 1,2 });
  FAST_CHECK_UNARY(J.isApprox(2 * xm.transpose()));
}
