/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#define EIGEN_RUNTIME_NO_MALLOC

#include "SolverTestFunctions.h"

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/function/abstract/LinearFunction.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/hint/Substitution.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/solver/defaultLeastSquareSolver.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/Proportional.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/VelocityDamper.h>

using namespace tvm;
using namespace Eigen;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

TEST_CASE("Basic problem") {}

TEST_CASE("Substitution")
{
  Space s1(2);
  int dim = 3;
  Space s2(dim);

  VectorXd ddx0;
  VectorXd ddq0;
  {
    VariablePtr x = s1.createVariable("x");
    VariablePtr dx = dot(x);
    x->set(Vector2d(0.5, 0.5));
    dx->set(Vector2d::Zero());

    VariablePtr q = s2.createVariable("q");
    VariablePtr dq = dot(q);
    q->set(Vector3d(0.4, -0.6, 0.9));
    dq->set(Vector3d::Zero());

    auto sf = std::make_shared<SphereFunction>(x, Vector2d(0, 0), 1);
    auto rf = std::make_shared<Simple2dRobotEE>(q, Vector2d(2, 0), Vector3d(1, 1, 1));
    auto idx = std::make_shared<function::IdentityFunction>(x);
    auto df = std::make_shared<Difference>(rf, idx);
    auto idq = std::make_shared<function::IdentityFunction>(dot(q, 2));

    VectorXd v(2);
    v << 0, 0;
    Vector3d b = Vector3d::Constant(1.5);

    double dt = 1e-1;
    LinearizedControlProblem lpb;
    auto t1 = lpb.add(sf == 0., task_dynamics::PD(2), {requirements::PriorityLevel(0)});
    auto t2 = lpb.add(df == v, task_dynamics::PD(2), {requirements::PriorityLevel(0)});
    auto t3 =
        lpb.add(-b <= q <= b, task_dynamics::VelocityDamper(dt, {1., 0.01, 0, 1}), {requirements::PriorityLevel(0)});
    auto t4 = lpb.add(idq == 0., task_dynamics::None(), {requirements::PriorityLevel(1)});

    scheme::WeightedLeastSquares solver(solver::DefaultLSSolverFactory{});
    tvm::utils::set_is_malloc_allowed(false);
    solver.solve(lpb);
    tvm::utils::set_is_malloc_allowed(true);
    ddx0 = dot(x, 2)->value();
    ddq0 = dot(q, 2)->value();
  }

  VectorXd ddxs;
  VectorXd ddqs;
  {
    VariablePtr x = s1.createVariable("x");
    VariablePtr dx = dot(x);
    x->set(Vector2d(0.5, 0.5));
    dx->set(Vector2d::Zero());

    VariablePtr q = s2.createVariable("q");
    VariablePtr dq = dot(q);
    q->set(Vector3d(0.4, -0.6, 0.9));
    dq->set(Vector3d::Zero());

    auto sf = std::make_shared<SphereFunction>(x, Vector2d(0, 0), 1);
    auto rf = std::make_shared<Simple2dRobotEE>(q, Vector2d(2, 0), Vector3d(1, 1, 1));
    auto idx = std::make_shared<function::IdentityFunction>(x);
    auto df = std::make_shared<Difference>(rf, idx);
    auto idq = std::make_shared<function::IdentityFunction>(dot(q, 2));

    VectorXd v(2);
    v << 0, 0;
    Vector3d b = Vector3d::Constant(1.5);

    double dt = 1e-1;
    LinearizedControlProblem lpb;
    auto t1 = lpb.add(sf == 0., task_dynamics::PD(2), {requirements::PriorityLevel(0)});
    auto t2 = lpb.add(df == v, task_dynamics::PD(2), {requirements::PriorityLevel(0)});
    auto t3 =
        lpb.add(-b <= q <= b, task_dynamics::VelocityDamper(dt, {1., 0.01, 0, 1}), {requirements::PriorityLevel(0)});
    auto t4 = lpb.add(idq == 0., task_dynamics::None(), {requirements::PriorityLevel(1)});

    lpb.add(hint::Substitution(lpb.constraint(*t2), dot(x, 2)));

    scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});
    tvm::utils::set_is_malloc_allowed(false);
    solver.solve(lpb);
    tvm::utils::set_is_malloc_allowed(true);
    ddxs = dot(x, 2)->value();
    ddqs = dot(q, 2)->value();
  }

  FAST_CHECK_UNARY(ddx0.isApprox(ddxs, 1e-5));
  FAST_CHECK_UNARY(ddq0.isApprox(ddqs, 1e-5));
}

#include <Eigen/Geometry>

Matrix3d hat(const VectorConstRef & v)
{
  Matrix3d H;
  H << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return H;
}

Quaterniond exp(const VectorConstRef & v)
{
  double t = v.norm();
  if(t < 1e-14)
    return Quaterniond::Identity();

  Vector3d u = std::sin(t / 2) * v / t;

  return {std::cos(t / 2), u.x(), u.y(), u.z()};
}

class SO3Action : public function::abstract::Function
{
public:
  SET_UPDATES(SO3Action, Rotation, Value, Jacobian)

  SO3Action(VariablePtr q, VariablePtr v) : function::abstract::Function(3), q_(*q), v_(*v)
  {
    registerUpdates(Update::Rotation, &SO3Action::updateRotation, Update::Value, &SO3Action::updateValue,
                    Update::Jacobian, &SO3Action::updateJacobian);
    addInternalDependency<SO3Action>(Update::Value, Update::Rotation);
    addInternalDependency<SO3Action>(Update::Jacobian, Update::Rotation);
    addOutputDependency<SO3Action>(Output::Value, Update::Value);
    addOutputDependency<SO3Action>(Output::Jacobian, Update::Jacobian);
    addVariable(q, false);
    addVariable(v, true);
  }

  void updateRotation()
  {
    const auto & d = q_.value();
    R_ = Eigen::Quaterniond(d[0], d[1], d[2], d[3]).toRotationMatrix();
  }

  void updateValue() { value_ = R_ * v_.value(); }

  void updateJacobian()
  {
    jacobian_.at(&q_) = -R_ * hat(v_.value());
    jacobian_.at(&v_) = R_;
  }

private:
  Variable & q_;
  Variable & v_;
  Matrix3d R_;
};

class SE3Action : public function::abstract::Function
{
public:
  SET_UPDATES(SE3Action, Rotation, Value, Jacobian)

  SE3Action(VariablePtr h, VariablePtr v) : function::abstract::Function(3), h_(*h), v_(*v)
  {
    registerUpdates(Update::Rotation, &SE3Action::updateRotation, Update::Value, &SE3Action::updateValue,
                    Update::Jacobian, &SE3Action::updateJacobian);
    addInternalDependency<SE3Action>(Update::Value, Update::Rotation);
    addInternalDependency<SE3Action>(Update::Jacobian, Update::Rotation);
    addOutputDependency<SE3Action>(Output::Value, Update::Value);
    addOutputDependency<SE3Action>(Output::Jacobian, Update::Jacobian);
    addVariable(h, false);
    addVariable(v, true);
  }

  void updateRotation()
  {
    const auto & d = h_.value();
    R_ = Eigen::Quaterniond(d[0], d[1], d[2], d[3]).toRotationMatrix();
  }

  void updateValue() { value_ = R_ * v_.value() + h_.value().tail<3>(); }

  void updateJacobian()
  {
    auto & Jh = jacobian_.at(&h_);
    Jh.leftCols<3>() = -R_ * hat(v_.value());
    Jh.rightCols<3>() = R_;
    jacobian_.at(&v_) = R_;
  }

private:
  Variable & h_;
  Variable & v_;
  Matrix3d R_;
};

TEST_CASE("Substitution with non-Euclidean variables")
{
  Space SO3(Space::Type::SO3);
  Space SE3(Space::Type::SE3);
  Space R3(3);

  VariablePtr h1 = SE3.createVariable("h1");
  h1 << 1, 0, 0, 0, 0, 0, 0;
  VariablePtr h2 = SE3.createVariable("h2");
  h2 << 1, 0, 0, 0, 0, 0, 0;
  VariablePtr v1 = R3.createVariable("v1");
  VariablePtr v2 = R3.createVariable("v2");

  VariablePtr h1r = h1->subvariable(SO3, "h1r");
  VariablePtr h1t = h1->subvariable(R3, "h1t", SO3);
  VariablePtr h2r = h2->subvariable(SO3, "h2r");
  VariablePtr h2t = h2->subvariable(R3, "h2t", SO3);

  auto fe1 = std::make_shared<SE3Action>(h1, v1);
  auto fe2 = std::make_shared<SE3Action>(h2, v2);

  auto sub12 = std::make_shared<Difference>(fe1, fe2);

  double dt = 0.1;
  auto updateSE3 = [&dt](VariablePtr h, const VectorConstRef & wv) {
    const auto & d = h->value();
    const auto & w = wv.head<3>();
    Quaterniond r = Quaterniond(d[0], d[1], d[2], d[3]) * exp(w * dt);
    h << r.w(), r.x(), r.y(), r.z(), d.tail<3>() + dt * wv.tail<3>();
  };
  auto updateR = [&dt](VariablePtr x, const VectorConstRef & dx) { x->set(x->value() + dt * dx); };

  {
    LinearizedControlProblem lpb;
    auto t1 = lpb.add(sub12 == 0., task_dynamics::Proportional(1), {requirements::PriorityLevel(0)});
    auto t2 = lpb.add(v1 == Vector3d(1, 0, 0), task_dynamics::Proportional(1), {requirements::PriorityLevel(0)});
    auto t3 = lpb.add(v2 == Vector3d(0, 1, 0), task_dynamics::Proportional(1), {requirements::PriorityLevel(0)});
    auto t4 = lpb.add(h1t == 0., task_dynamics::Proportional(1), {requirements::PriorityLevel(1)});
    auto t5 = lpb.add(h2t == 0., task_dynamics::Proportional(1), {requirements::PriorityLevel(1)});

    lpb.add(hint::Substitution(lpb.constraint(*t1), dot(h1t)));

    scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});
    for(int i = 0; i < 200; ++i)
    {
      solver.solve(lpb);

      updateSE3(h1, dot(h1)->value());
      updateSE3(h2, dot(h2)->value());
      updateR(v1, dot(v1)->value());
      updateR(v2, dot(v2)->value());
    }

    FAST_CHECK_LE(sub12->value().norm(), 1e-8);
    FAST_CHECK_UNARY(v1->value().isApprox(Vector3d::UnitX(), 1e-8));
    FAST_CHECK_UNARY(v2->value().isApprox(Vector3d::UnitY(), 1e-8));
    FAST_CHECK_LE(h1t->value().norm(), 1e-8);
    FAST_CHECK_LE(h2t->value().norm(), 1e-8);
  }

  {
    LinearizedControlProblem lpb;
    auto t1 = lpb.add(sub12 == 0., task_dynamics::Proportional(1), {requirements::PriorityLevel(0)});
    auto t2 = lpb.add(v1 == Vector3d(1, 0, 0), task_dynamics::Proportional(1), {requirements::PriorityLevel(0)});
    auto t3 = lpb.add(v2 == Vector3d(0, 1, 0), task_dynamics::Proportional(1), {requirements::PriorityLevel(0)});
    auto t4 = lpb.add(h1t == 0., task_dynamics::Proportional(1), {requirements::PriorityLevel(1)});
    auto t5 = lpb.add(h2t == 0., task_dynamics::Proportional(1), {requirements::PriorityLevel(1)});

    lpb.add(hint::Substitution(lpb.constraint(*t1), dot(h2r)));

    scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions{});
    for(int i = 0; i < 200; ++i)
    {
      solver.solve(lpb);

      updateSE3(h1, dot(h1)->value());
      updateSE3(h2, dot(h2)->value());
      updateR(v1, dot(v1)->value());
      updateR(v2, dot(v2)->value());
    }

    FAST_CHECK_LE(sub12->value().norm(), 1e-8);
    FAST_CHECK_UNARY(v1->value().isApprox(Vector3d::UnitX(), 1e-8));
    FAST_CHECK_UNARY(v2->value().isApprox(Vector3d::UnitY(), 1e-8));
    FAST_CHECK_LE(h1t->value().norm(), 1e-8);
    FAST_CHECK_LE(h2t->value().norm(), 1e-8);
  }
}
