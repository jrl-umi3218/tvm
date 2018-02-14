#include "SolverTestFunctions.h"

#include <iostream>

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/function/abstract/LinearFunction.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/VelocityDamper.h>

using namespace tvm;
using namespace Eigen;

void checkJacobian(FunctionPtr f)
{
  auto userValue = std::make_shared<graph::internal::Inputs>();
  userValue->addInput(f, tvm::internal::FirstOrderProvider::Output::Value);
  auto userFull = std::make_shared<graph::internal::Inputs>();
  userFull->addInput(f, tvm::internal::FirstOrderProvider::Output::Value);
  userFull->addInput(f, tvm::internal::FirstOrderProvider::Output::Jacobian);

  tvm::graph::CallGraph gValue;
  gValue.add(userValue);
  gValue.update();
  tvm::graph::CallGraph gFull;
  gFull.add(userFull);
  gFull.update();

  for (auto x : f->variables())
  {
    VectorXd x0 = VectorXd::Random(x->size());
    x->value(x0);
    gFull.execute();
    auto f0 = f->value();
    auto J0 = f->jacobian(*x);

    const double h = 1e-6;
    MatrixXd J(f->size(), x->size());
    for (int i = 0; i < x->size(); ++i)
    {
      auto xi = x0;
      xi[i] += h;
      x->value(xi);
      gValue.execute();
      J.col(i) = (f->value() - f0) / h;
    }

    std::cout << x->name() << ": " << (J - J0).array().abs().maxCoeff() << std::endl;
  }
}

void checkNormalAcc(FunctionPtr f)
{
  auto user1stOrder = std::make_shared<graph::internal::Inputs>();
  user1stOrder->addInput(f, tvm::internal::FirstOrderProvider::Output::Value);
  user1stOrder->addInput(f, tvm::internal::FirstOrderProvider::Output::Jacobian);
  auto userFull = std::make_shared<graph::internal::Inputs>();
  userFull->addInput(f, tvm::internal::FirstOrderProvider::Output::Value);
  userFull->addInput(f, tvm::internal::FirstOrderProvider::Output::Jacobian);
  userFull->addInput(f, function::abstract::Function::Output::Velocity);
  userFull->addInput(f, function::abstract::Function::Output::NormalAcceleration);

  tvm::graph::CallGraph g1stOrder;
  g1stOrder.add(user1stOrder);
  g1stOrder.update();
  tvm::graph::CallGraph gFull;
  gFull.add(userFull);
  gFull.update();

  VectorXd na = VectorXd::Zero(f->size());
  VectorXd na0;
  for (auto x : f->variables())
  {
    VariablePtr v = dot(x);
    VectorXd x0 = VectorXd::Random(x->size());
    VectorXd v0 = VectorXd::Random(x->size());
    x->value(x0);
    v->value(v0);
    gFull.execute();
    auto J0 = f->jacobian(*x);
    na0 = f->normalAcceleration();

    const double h = 1e-6;
    x->value(x0 + h*v0);
    g1stOrder.execute();
    MatrixXd J1 = f->jacobian(*x);
    na += (J1 - J0) / h * v0;
  }

  std::cout << (na - na0).array().abs().maxCoeff() << std::endl;
}

void solverTest01()
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");
  VariablePtr dx = dot(x);
  x->value(Vector2d(0.5,0.5));
  dx->value(Vector2d::Zero());

  int dim = 3;
  Space s2(dim);
  VariablePtr q = s2.createVariable("q");
  VariablePtr dq = dot(q);
  q->value(Vector3d(0.4, -0.6, 0.9));
  dq->value(Vector3d::Zero());

  auto sf = std::make_shared<SphereFunction>(x, Vector2d(0, 0), 1);
  auto rf = std::make_shared<Simple2dRobotEE>(q, Vector2d(2, 0), Vector3d(1, 1, 1));
  auto idx = std::make_shared<function::IdentityFunction>(x);
  auto df = std::make_shared<Difference>(rf, idx);

  VectorXd v(2); v << 0, 0;
  Vector3d b = Vector3d::Constant(1.5);

  double dt = 1e-1;
  ControlProblem pb(dt);
  LinearizedControlProblem lpb(pb);
  auto t1 = lpb.add(sf == 0., task_dynamics::PD(2), { requirements::PriorityLevel(0) });
  auto t2 = lpb.add(df == v, task_dynamics::PD(2), { requirements::PriorityLevel(0) });
  auto t3 = lpb.add(-b <= q <= b, task_dynamics::VelocityDamper(dt, { 1., 0.01, 0, 1 }), { requirements::PriorityLevel(0) });
  std::cout << t1->task.taskDynamics<task_dynamics::PD>()->gains().first << std::endl;

  scheme::WeightedLeastSquares solver;
  solver.solve(lpb);
  std::cout << "ddx = " << dot(x, 2)->value().transpose() << std::endl;
  std::cout << "ddq = " << dot(q, 2)->value().transpose() << std::endl;
}

void solverTest02()
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");

  Space s2(3);
  VariablePtr q = s2.createVariable("q");

  auto idx = std::make_shared<function::IdentityFunction>(x);
  auto idq = std::make_shared<function::IdentityFunction>(q);

  ControlProblem pb(0.1);
  pb.add(idx >= 0., task_dynamics::None(), { requirements::PriorityLevel(0) });
  pb.add(idq >= 0., task_dynamics::None(), { requirements::PriorityLevel(0) });

  LinearizedControlProblem lpb(pb);

  scheme::WeightedLeastSquares solver;
  solver.solve(lpb);
}

void solverTest03()
{
  VariablePtr x = Space(3).createVariable("x");
  VariablePtr q = Space(3).createVariable("q");

  Eigen::MatrixXd x_cstr_A(3,3); x_cstr_A.setZero();
  x_cstr_A(0,0) = 1;
  auto x_cstr = std::make_shared<function::BasicLinearFunction>(x_cstr_A, x);
  auto x_id = std::make_shared<function::IdentityFunction>(x);
  Eigen::VectorXd xT(3);
  xT << 100, 1, 2;
  Eigen::VectorXd qL(3);
  qL << -5, -5, -5;
  Eigen::VectorXd qU = -qL;

  ControlProblem pb(0.1);
  pb.add(x_cstr == 0., task_dynamics::PD(1.), { requirements::PriorityLevel(0) });
  pb.add(qL <= q <= qU, task_dynamics::None(), { requirements::PriorityLevel(0) });
  pb.add(x_id == xT, task_dynamics::PD(1.), { requirements::PriorityLevel(1) });

  LinearizedControlProblem lpb(pb);

  scheme::WeightedLeastSquares solver(true);
  auto dx = dot(x);
  dx->value(Eigen::VectorXd::Zero(3));
  for(int i = 0; i < 100; ++i)
  {
    solver.solve(lpb);
    auto x_v = x->value();
    auto dx_v = dx->value();
    dx_v += 0.005*dot(x,2)->value();
    x_v += 0.005*dx_v + 0.0005*0.005*dot(x,2)->value()/2;
    dx->value(dx_v);
    x->value(x_v);
  }
  std::cout << "Final x " << x->value().transpose() << std::endl;
  std::cout << "Final q " << q->value().transpose() << std::endl;
}
