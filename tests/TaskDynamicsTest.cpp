/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include "SolverTestFunctions.h"

#include <tvm/Variable.h>
#include <tvm/function/BasicLinearFunction.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/task_dynamics/Clamped.h>
#include <tvm/task_dynamics/Constant.h>
#include <tvm/task_dynamics/FeedForward.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/OneStepToZero.h>
#include <tvm/task_dynamics/Proportional.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/Reference.h>
#include <tvm/task_dynamics/VelocityDamper.h>
#include <tvm/utils/graph.h>

#include <Eigen/QR>

#include <iostream>

#if defined __i386__ || defined __aarch64__
#  define APPROX_I386(x) doctest::Approx(x)
#else
#  define APPROX_I386(x) x
#endif

using namespace Eigen;
using namespace tvm;

/** Feed forward value provider */
class FFProvider : public tvm::graph::abstract::Node<FFProvider>
{
public:
  SET_OUTPUTS(FFProvider, Value)
  SET_UPDATES(FFProvider, Value)

  FFProvider(const Eigen::VectorXd & value)
  {
    registerUpdates(Update::Value, &FFProvider::updateValue);
    addOutputDependency(Output::Value, Update::Value);
    value_ = value;
  }

  const Eigen::VectorXd & value() const { return value_; }

  void updateValue() { updated_ = true; }

  Eigen::VectorXd value_;
  bool updated_ = false;
};

TEST_CASE("Valid construction")
{
  VariablePtr x = Space(3).createVariable("x");
  x << Vector3d::Zero();
  auto f = std::make_shared<function::IdentityFunction>(x);
  task_dynamics::Constant td;
  CHECK_NOTHROW(td.impl(f, constraint::Type::EQUAL, Vector3d(1, 0, 0)));
  CHECK_THROWS(td.impl(f, constraint::Type::EQUAL, Vector2d(1, 0)));
}

TEST_CASE("Test Constant")
{
  VariablePtr x = Space(3).createVariable("x");
  x << Vector3d::Zero();
  auto f = std::make_shared<function::IdentityFunction>(x);
  task_dynamics::Constant td;
  auto tdi = td.impl(f, constraint::Type::EQUAL, Vector3d(1, 0, 0));

  tdi->updateValue();

  FAST_CHECK_EQ(td.order(), task_dynamics::Order::Zero);
  FAST_CHECK_EQ(tdi->order(), task_dynamics::Order::Zero);
  FAST_CHECK_UNARY(tdi->value().isApprox(Vector3d(1, 0, 0)));
  FAST_CHECK_UNARY(tdi->checkType<task_dynamics::Constant::Impl>());
  FAST_CHECK_UNARY_FALSE(tdi->checkType<task_dynamics::None::Impl>());
}

TEST_CASE("Test None")
{
  VariablePtr x = Space(3).createVariable("x");
  x << 1, 2, 3;
  MatrixXd A = MatrixXd::Random(2, 3);
  Vector2d b(1, 2);
  auto f = std::make_shared<function::BasicLinearFunction>(A, x, b);
  task_dynamics::None td;
  auto tdi = td.impl(f, constraint::Type::EQUAL, Vector2d(1, 0));

  f->updateValue();
  tdi->updateValue();

  FAST_CHECK_EQ(td.order(), task_dynamics::Order::Zero);
  FAST_CHECK_EQ(tdi->order(), task_dynamics::Order::Zero);
  FAST_CHECK_UNARY(tdi->value().isApprox(Vector2d(0, -2)));
  FAST_CHECK_UNARY(tdi->checkType<task_dynamics::None::Impl>());
  FAST_CHECK_UNARY_FALSE(tdi->checkType<task_dynamics::Constant::Impl>());
}

TEST_CASE("Test Proportional")
{
  VariablePtr x = Space(3).createVariable("x");
  x << 1, 2, 3;
  auto f = std::make_shared<SphereFunction>(x, Vector3d(1, 0, -3), 2);

  double kp = 2;
  task_dynamics::P td(kp);
  VectorXd rhs(1);
  rhs[0] = -1;
  auto tdi = td.impl(f, constraint::Type::EQUAL, rhs);

  f->updateValue();
  tdi->updateValue();

  FAST_CHECK_EQ(td.order(), task_dynamics::Order::One);
  FAST_CHECK_EQ(tdi->order(), task_dynamics::Order::One);
  FAST_CHECK_EQ(tdi->value()[0], -kp * (36 - rhs[0])); // -kp*||(1,2,3) - (1,0,-3)||^2 - 2^2 - rhs
  FAST_CHECK_UNARY(tdi->checkType<task_dynamics::P::Impl>());
  FAST_CHECK_UNARY_FALSE(tdi->checkType<task_dynamics::Constant::Impl>());

  kp = 3;
  dynamic_cast<task_dynamics::P::Impl *>(tdi.get())->gain(kp);
  tdi->updateValue();
  FAST_CHECK_EQ(tdi->value()[0], -kp * (36 - rhs[0])); // -kp*||(1,2,3) - (1,0,-3)||^2 - 2^2 - rhs
}

TEST_CASE("Test Proportional gain types")
{
  VariablePtr x = Space(3).createVariable("x");
  x << 1, 2, 3;
  MatrixXd A = MatrixXd::Random(3, 3);
  Vector3d b(1, 2, 3);
  auto f = std::make_shared<function::BasicLinearFunction>(A, x, b);

  double kps = 2;
  VectorXd kpv = Vector3d::Constant(2);
  MatrixXd kpm = 2 * Matrix3d::Identity();
  task_dynamics::P tds(kps);
  task_dynamics::P tdv(kpv);
  task_dynamics::P tdm(kpm);
  Vector3d rhs(-1, -2, -3);

  auto tdsi = tds.impl(f, constraint::Type::EQUAL, rhs);
  auto tdvi = tdv.impl(f, constraint::Type::EQUAL, rhs);
  auto tdmi = tdm.impl(f, constraint::Type::EQUAL, rhs);

  f->updateValue();
  tdsi->updateValue();
  tdvi->updateValue();
  tdmi->updateValue();

  FAST_CHECK_UNARY(tdsi->value().isApprox(tdvi->value()));
  FAST_CHECK_UNARY(tdsi->value().isApprox(tdmi->value()));

  kps = 3;
  kpv = Vector3d::Constant(3);
  kpm = 3 * Matrix3d::Identity();

  dynamic_cast<task_dynamics::P::Impl *>(tdsi.get())->gain(kps);
  dynamic_cast<task_dynamics::P::Impl *>(tdvi.get())->gain(kpv);
  dynamic_cast<task_dynamics::P::Impl *>(tdmi.get())->gain(kpm);

  tdsi->updateValue();
  tdvi->updateValue();
  tdmi->updateValue();

  FAST_CHECK_UNARY(tdsi->value().isApprox(tdvi->value()));
  FAST_CHECK_UNARY(tdsi->value().isApprox(tdmi->value()));

  dynamic_cast<task_dynamics::P::Impl *>(tdsi.get())->gain() = 5;
  dynamic_cast<task_dynamics::P::Impl *>(tdvi.get())->gain<VectorXd>()[0] = 5;
  dynamic_cast<task_dynamics::P::Impl *>(tdvi.get())->gain<VectorXd>()[1] = 5;
  dynamic_cast<task_dynamics::P::Impl *>(tdvi.get())->gain<VectorXd>()[2] = 5;
  dynamic_cast<task_dynamics::P::Impl *>(tdmi.get())->gain<MatrixXd>().diagonal() = Vector3d::Constant(5);

  tdsi->updateValue();
  tdvi->updateValue();
  tdmi->updateValue();

  FAST_CHECK_UNARY(tdsi->value().isApprox(tdvi->value()));
  FAST_CHECK_UNARY(tdsi->value().isApprox(tdmi->value()));
  CHECK_THROWS(dynamic_cast<task_dynamics::P::Impl *>(tdvi.get())->gain<MatrixXd>());
}

TEST_CASE("Test Proportional Derivative")
{
  VariablePtr x = Space(3).createVariable("x");
  VariablePtr dx = dot(x);
  x << 1, 2, 3;
  dx << 1, 1, 1;
  auto f = std::make_shared<SphereFunction>(x, Vector3d(1, 0, -3), 2);

  double kp = 2;
  double kv = 3;
  task_dynamics::PD td(kp, kv);
  VectorXd rhs(1);
  rhs[0] = -1;
  auto tdi = td.impl(f, constraint::Type::EQUAL, rhs);

  f->updateValue();
  f->updateVelocityAndNormalAcc();
  tdi->updateValue();

  FAST_CHECK_EQ(td.order(), task_dynamics::Order::Two);
  FAST_CHECK_EQ(tdi->order(), task_dynamics::Order::Two);
  FAST_CHECK_EQ(tdi->value()[0], -kp * (36 - rhs[0]) - kv * 16);
  FAST_CHECK_UNARY(tdi->checkType<task_dynamics::PD::Impl>());
  FAST_CHECK_UNARY_FALSE(tdi->checkType<task_dynamics::Constant::Impl>());
}

TEST_CASE("Test Feed Forward Proportional Derivative")
{
  VariablePtr x = Space(3).createVariable("x");
  VariablePtr dx = dot(x);
  x << 1, 2, 3;
  dx << 1, 1, 1;
  auto f = std::make_shared<SphereFunction>(x, Vector3d(1, 0, -3), 2);

  auto provider = std::make_shared<FFProvider>(Eigen::VectorXd::Constant(1, 10.0));

  double kp = 2;
  double kv = 3;
  task_dynamics::FeedForwardPD td(provider, &FFProvider::value, FFProvider::Output::Value, kp, kv);
  VectorXd rhs(1);
  rhs[0] = -1;
  TaskDynamicsPtr tdi = td.impl(f, constraint::Type::EQUAL, rhs);

  auto Value = task_dynamics::abstract::TaskDynamicsImpl::Output::Value;
  auto gl = utils::generateUpdateGraph(tdi, Value);
  gl->execute();

  FAST_CHECK_UNARY(provider->updated_);
  FAST_CHECK_EQ(td.order(), task_dynamics::Order::Two);
  FAST_CHECK_EQ(tdi->order(), task_dynamics::Order::Two);
  FAST_CHECK_EQ(tdi->value()[0], -kp * (36 - rhs[0]) - kv * 16 + 10.0);
  FAST_CHECK_UNARY(tdi->checkType<task_dynamics::PD::Impl>());
  FAST_REQUIRE_UNARY(tdi->checkType<task_dynamics::FeedForwardPD::Impl>()); // REQUIRE because we do a static_cast after
  FAST_CHECK_UNARY_FALSE(tdi->checkType<task_dynamics::Constant::Impl>());

  kp = 3;
  kv = 2;
  static_cast<task_dynamics::FeedForwardPD::Impl *>(tdi.get())->gains(kp, kv);
  gl->execute();
  FAST_CHECK_EQ(tdi->value()[0], -kp * (36 - rhs[0]) - kv * 16 + 10.0);
}

TEST_CASE("Test Proportional Derivative gain types")
{
  VariablePtr x = Space(3).createVariable("x");
  VariablePtr dx = dot(x);
  x << 1, 2, 3;
  dx << 1, 1, 1;
  auto f = std::make_shared<Simple2dRobotEE>(x, Vector2d(0, 0), Vector3d::Ones());

  double kps = 2;
  double kvs = 3;
  VectorXd kpv = Vector2d::Constant(2);
  VectorXd kvv = Vector2d::Constant(3);
  MatrixXd kpm = 2 * Matrix2d::Identity();
  MatrixXd kvm = 3 * Matrix2d::Identity();
  task_dynamics::PD tdss(kps, kvs);
  task_dynamics::PD tdsv(kps, kvv);
  task_dynamics::PD tdsm(kps, kvm);
  task_dynamics::PD tdvs(kpv, kvs);
  task_dynamics::PD tdvv(kpv, kvv);
  task_dynamics::PD tdvm(kpv, kvm);
  task_dynamics::PD tdms(kpm, kvs);
  task_dynamics::PD tdmv(kpm, kvv);
  task_dynamics::PD tdmm(kpm, kvm);
  task_dynamics::PD tds(kvs);
  task_dynamics::PD tdv(kvv);
  task_dynamics::PD tdm(kvm);
  Vector2d rhs(-1, -2);
  auto tdssi = tdss.impl(f, constraint::Type::EQUAL, rhs);
  auto tdsvi = tdsv.impl(f, constraint::Type::EQUAL, rhs);
  auto tdsmi = tdsm.impl(f, constraint::Type::EQUAL, rhs);
  auto tdvsi = tdvs.impl(f, constraint::Type::EQUAL, rhs);
  auto tdvvi = tdvv.impl(f, constraint::Type::EQUAL, rhs);
  auto tdvmi = tdvm.impl(f, constraint::Type::EQUAL, rhs);
  auto tdmsi = tdms.impl(f, constraint::Type::EQUAL, rhs);
  auto tdmvi = tdmv.impl(f, constraint::Type::EQUAL, rhs);
  auto tdmmi = tdmm.impl(f, constraint::Type::EQUAL, rhs);
  auto tdsi = tds.impl(f, constraint::Type::EQUAL, rhs);
  auto tdvi = tdv.impl(f, constraint::Type::EQUAL, rhs);
  auto tdmi = tdm.impl(f, constraint::Type::EQUAL, rhs);

  f->updateValue();
  f->updateVelocityAndNormalAcc();
  tdssi->updateValue();
  tdsvi->updateValue();
  tdsmi->updateValue();
  tdvsi->updateValue();
  tdvvi->updateValue();
  tdvmi->updateValue();
  tdmsi->updateValue();
  tdmvi->updateValue();
  tdmmi->updateValue();
  tdsi->updateValue();
  tdvi->updateValue();
  tdmi->updateValue();

  FAST_CHECK_UNARY(tdssi->value().isApprox(tdsvi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdsvi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdvsi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdvvi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdvmi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdmsi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdmvi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdmmi->value()));
  FAST_CHECK_UNARY(tdsi->value().isApprox(tdvi->value()));
  FAST_CHECK_UNARY(tdsi->value().isApprox(tdvi->value()));

  kps = 3;
  kvs = 1;
  kpv = Vector2d::Constant(3);
  kvv = Vector2d::Constant(1);
  kpm = 3 * Matrix2d::Identity();
  kvm = 1 * Matrix2d::Identity();
  dynamic_cast<task_dynamics::PD::Impl *>(tdssi.get())->gains(kps, kvs);
  dynamic_cast<task_dynamics::PD::Impl *>(tdsvi.get())->gains(kps, kvv);
  dynamic_cast<task_dynamics::PD::Impl *>(tdsmi.get())->gains(kps, kvm);
  dynamic_cast<task_dynamics::PD::Impl *>(tdvsi.get())->gains(kpv, kvs);
  dynamic_cast<task_dynamics::PD::Impl *>(tdvvi.get())->gains(kpv, kvv);
  dynamic_cast<task_dynamics::PD::Impl *>(tdvmi.get())->gains(kpv, kvm);
  dynamic_cast<task_dynamics::PD::Impl *>(tdmsi.get())->gains(kpm, kvs);
  dynamic_cast<task_dynamics::PD::Impl *>(tdmvi.get())->gains(kpm, kvv);
  dynamic_cast<task_dynamics::PD::Impl *>(tdmmi.get())->gains(kpm, kvm);
  dynamic_cast<task_dynamics::PD::Impl *>(tdsi.get())->gains(kps);
  dynamic_cast<task_dynamics::PD::Impl *>(tdvi.get())->gains(kpv);
  dynamic_cast<task_dynamics::PD::Impl *>(tdmi.get())->gains(kpm);

  tdssi->updateValue();
  tdsvi->updateValue();
  tdsmi->updateValue();
  tdvsi->updateValue();
  tdvvi->updateValue();
  tdvmi->updateValue();
  tdmsi->updateValue();
  tdmvi->updateValue();
  tdmmi->updateValue();
  tdsi->updateValue();
  tdvi->updateValue();
  tdmi->updateValue();

  FAST_CHECK_UNARY(tdssi->value().isApprox(tdsvi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdsvi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdvsi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdvvi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdvmi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdmsi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdmvi->value()));
  FAST_CHECK_UNARY(tdssi->value().isApprox(tdmmi->value()));
  FAST_CHECK_UNARY(tdsi->value().isApprox(tdvi->value()));
  FAST_CHECK_UNARY(tdsi->value().isApprox(tdvi->value()));

  dynamic_cast<task_dynamics::PD::Impl *>(tdssi.get())->kp() = 5;
  dynamic_cast<task_dynamics::PD::Impl *>(tdssi.get())->kv() = 2;
  dynamic_cast<task_dynamics::PD::Impl *>(tdvmi.get())->kp<VectorXd>() << 5, 5;
  dynamic_cast<task_dynamics::PD::Impl *>(tdvmi.get())->kv<MatrixXd>() = 2 * Matrix2d::Identity();

  tdssi->updateValue();
  tdvmi->updateValue();

  FAST_CHECK_UNARY(tdssi->value().isApprox(tdvmi->value()));
}

TEST_CASE("Test Velocity Damper")
{
  VariablePtr x = Space(3).createVariable("x");
  VariablePtr dx = dot(x);
  auto f = std::make_shared<function::IdentityFunction>(x);

  double di = 3;
  double ds = 1;
  double xsi = 2;
  double dt = 0.1;

  // test validity
  CHECK_THROWS(task_dynamics::VelocityDamper({0.5, ds, xsi}));
  CHECK_THROWS(task_dynamics::VelocityDamper(dt, {0.5, ds, xsi}, 1000));
  CHECK_THROWS(task_dynamics::VelocityDamper({di, ds, -1}));
  CHECK_THROWS(task_dynamics::VelocityDamper(dt, {di, ds, -1}, 1000));
  CHECK_THROWS(task_dynamics::VelocityDamper(0, {di, ds, xsi}, 1000));
  CHECK_THROWS(task_dynamics::VelocityDamper(-0.1, {di, ds, xsi}, 1000));

  // test kinematics
  task_dynamics::VelocityDamper td1({di, ds, xsi});
  {
    x << 1, 2, 4;
    auto tdl = td1.impl(f, constraint::Type::GREATER_THAN, Vector3d::Zero());

    f->updateValue();
    tdl->updateValue();

    FAST_CHECK_EQ(td1.order(), task_dynamics::Order::One);
    FAST_CHECK_EQ(tdl->order(), task_dynamics::Order::One);
    FAST_CHECK_EQ(tdl->value()[0], 0);
    FAST_CHECK_EQ(tdl->value()[1], -1);
    FAST_CHECK_EQ(tdl->value()[2], -constant::big_number);
    FAST_CHECK_UNARY(tdl->checkType<task_dynamics::VelocityDamper::Impl>());
    FAST_CHECK_UNARY_FALSE(tdl->checkType<task_dynamics::Constant::Impl>());

    x << -1, -2, -4;
    auto tdu = td1.impl(f, constraint::Type::LOWER_THAN, Vector3d::Zero());
    f->updateValue();
    tdu->updateValue();
    FAST_CHECK_EQ(tdu->value()[0], 0);
    FAST_CHECK_EQ(tdu->value()[1], 1);
    FAST_CHECK_EQ(tdu->value()[2], constant::big_number);
  }

  // test dynamics
  double big = 1000;
  task_dynamics::VelocityDamper td2(dt, {di, ds, xsi}, big);
  {
    x << 1, 2, 4;
    dx << 1, 1, 1;
    auto tdl = td2.impl(f, constraint::Type::GREATER_THAN, Vector3d::Zero());

    f->updateValue();
    f->updateVelocity();
    tdl->updateValue();

    FAST_CHECK_EQ(td2.order(), task_dynamics::Order::Two);
    FAST_CHECK_EQ(tdl->order(), task_dynamics::Order::Two);
    FAST_CHECK_UNARY(tdl->value().isApprox(Vector3d(-10, -20, -1000)));
    FAST_CHECK_UNARY(tdl->checkType<task_dynamics::VelocityDamper::Impl>());
    FAST_CHECK_UNARY_FALSE(tdl->checkType<task_dynamics::Constant::Impl>());

    x << -1, -2, -4;
    dx << -1, -1, -1;
    auto tdu = td2.impl(f, constraint::Type::LOWER_THAN, Vector3d::Zero());

    f->updateValue();
    f->updateVelocity();
    tdu->updateValue();
    FAST_CHECK_UNARY(tdu->value().isApprox(Vector3d(10, 20, 1000)));
  }
}

TEST_CASE("Test Anisotropic Velocity Damper")
{
  VariablePtr x = Space(3).createVariable("x");
  VariablePtr dx = dot(x);
  auto f = std::make_shared<function::IdentityFunction>(x);

  Eigen::Vector3d di = {3, 6, 1};
  Eigen::Vector3d ds = {1, 1, 0.1};
  Eigen::Vector3d xsi = {2, 1, 0.5};
  double dt = 0.1;

  // test validity
  // Should throw because ds > di
  CHECK_THROWS(task_dynamics::VelocityDamper({ds, di, xsi}));
  // Should throw because xsi dimension don't match the others
  CHECK_THROWS(task_dynamics::VelocityDamper({di, ds, Eigen::Vector4d::Zero()}));
  // Should throw because negative damping
  CHECK_THROWS(task_dynamics::VelocityDamper({di, ds, Eigen::Vector3d::Constant(-1)}));
  // Same 3 checks with dt parameter
  CHECK_THROWS(task_dynamics::VelocityDamper(dt, {ds, di, xsi}));
  CHECK_THROWS(task_dynamics::VelocityDamper(dt, {di, ds, Eigen::Vector4d::Zero()}));
  CHECK_THROWS(task_dynamics::VelocityDamper(dt, {di, ds, Eigen::Vector3d::Constant(-1)}));
  // Should throw because dt <= 0
  CHECK_THROWS(task_dynamics::VelocityDamper(0, {di, ds, xsi}));
  CHECK_THROWS(task_dynamics::VelocityDamper(-0.1, {di, ds, xsi}));

  // test kinematics
  task_dynamics::VelocityDamper td1({di, ds, xsi});
  {
    x << 1, 2, 4;
    auto tdl = td1.impl(f, constraint::Type::GREATER_THAN, Vector3d::Zero());

    f->updateValue();
    tdl->updateValue();

    FAST_CHECK_EQ(td1.order(), task_dynamics::Order::One);
    FAST_CHECK_EQ(tdl->order(), task_dynamics::Order::One);
    FAST_CHECK_EQ(tdl->value()[0], 0);
    FAST_CHECK_EQ(tdl->value()[1], -0.2);
    FAST_CHECK_EQ(tdl->value()[2], -constant::big_number);
    FAST_CHECK_UNARY(tdl->checkType<task_dynamics::VelocityDamper::Impl>());
    FAST_CHECK_UNARY_FALSE(tdl->checkType<task_dynamics::Constant::Impl>());

    x << -1, -2, -4;
    auto tdu = td1.impl(f, constraint::Type::LOWER_THAN, Vector3d::Zero());
    f->updateValue();
    tdu->updateValue();
    FAST_CHECK_EQ(tdu->value()[0], 0);
    FAST_CHECK_EQ(tdu->value()[1], 0.2);
    FAST_CHECK_EQ(tdu->value()[2], constant::big_number);
  }

  // test dynamics
  double big = 1000;
  task_dynamics::VelocityDamper td2(dt, {di, ds, xsi}, big);
  {
    x << 1, 2, 4;
    dx << 1, 1, 1;
    auto tdl = td2.impl(f, constraint::Type::GREATER_THAN, Vector3d::Zero());

    f->updateValue();
    f->updateVelocity();
    tdl->updateValue();

    FAST_CHECK_EQ(td2.order(), task_dynamics::Order::Two);
    FAST_CHECK_EQ(tdl->order(), task_dynamics::Order::Two);
    FAST_CHECK_UNARY(tdl->value().isApprox(Vector3d(-10, -12, -1000)));
    FAST_CHECK_UNARY(tdl->checkType<task_dynamics::VelocityDamper::Impl>());
    FAST_CHECK_UNARY_FALSE(tdl->checkType<task_dynamics::Constant::Impl>());

    x << -1, -2, -4;
    dx << -1, -1, -1;
    auto tdu = td2.impl(f, constraint::Type::LOWER_THAN, Vector3d::Zero());

    f->updateValue();
    f->updateVelocity();
    tdu->updateValue();
    FAST_CHECK_UNARY(tdu->value().isApprox(Vector3d(10, 12, 1000)));
  }
}

TEST_CASE("Test automatic xsi")
{
  VariablePtr x = Space(3).createVariable("x");
  VariablePtr dx = dot(x);
  auto f = std::make_shared<function::IdentityFunction>(x);

  double di = 3;
  double ds = 1;
  double xsiOff = 1;

  // test kinematics
  double big = 100;
  task_dynamics::VelocityDamper td1({di, ds, 0, xsiOff}, big);
  {
    x << 5, 4, 2;
    dx << -0.5, -0.5, -0.5;
    TaskDynamicsPtr tdl = td1.impl(f, constraint::Type::GREATER_THAN, Vector3d::Zero());

    auto Value = task_dynamics::abstract::TaskDynamicsImpl::Output::Value;
    auto gl = utils::generateUpdateGraph(tdl, Value);

    gl->execute();
    FAST_CHECK_UNARY(tdl->value().isApprox(Vector3d(-big, -big, -1)));

    x << 4.5, 3.5, 1.5;
    gl->execute();
    FAST_CHECK_UNARY(tdl->value().isApprox(Vector3d(-big, -big, -0.5)));

    x << 4, 3, 1;
    gl->execute();
    FAST_CHECK_UNARY(tdl->value().isApprox(Vector3d(-big, -1.5, 0)));

    // we check that two consecutive updates with the same variable values give the same results.
    gl->execute();
    FAST_CHECK_UNARY(tdl->value().isApprox(Vector3d(-big, -1.5, 0)));

    dx << -0.5, -0.5, 0;
    x << 3.5, 2.5, 1;
    gl->execute();
    FAST_CHECK_UNARY(tdl->value().isApprox(Vector3d(-big, -1.125, 0)));

    x << 3, 2, 1;
    gl->execute();
    FAST_CHECK_UNARY(tdl->value().isApprox(Vector3d(-1.5, -0.75, 0)));

    x << -5, -4, -2;
    dx << 0.5, 0.5, 0.5;
    TaskDynamicsPtr tdu = td1.impl(f, constraint::Type::LOWER_THAN, Vector3d::Zero());

    auto gu = utils::generateUpdateGraph(tdu, Value);

    gu->execute();
    FAST_CHECK_UNARY(tdu->value().isApprox(Vector3d(big, big, 1)));

    x << -4.5, -3.5, -1.5;
    gu->execute();
    FAST_CHECK_UNARY(tdu->value().isApprox(Vector3d(big, big, 0.5)));

    x << -4, -3, -1;
    gu->execute();
    FAST_CHECK_UNARY(tdu->value().isApprox(Vector3d(big, 1.5, 0)));

    dx << 0.5, 0.5, 0;
    x << -3.5, -2.5, -1;
    gu->execute();
    FAST_CHECK_UNARY(tdu->value().isApprox(Vector3d(big, 1.125, 0)));

    x << -3, -2, -1;
    gu->execute();
    FAST_CHECK_UNARY(tdu->value().isApprox(Vector3d(1.5, 0.75, 0)));
  }
}

TEST_CASE("Test Reference")
{
  VariablePtr x = Space(3).createVariable("x");
  VariablePtr dx = dot(x);
  x << 1, 2, 3;
  dx << 1, 1, 1;
  auto f = std::make_shared<Simple2dRobotEE>(x, Vector2d(0, 0), Vector3d::Ones());

  VariablePtr u = Space(1).createVariable("u");
  u << 0;
  MatrixXd A(2, 1);
  A << 2, 1;
  VectorXd b(2);
  b << -1, 0;
  auto ref = std::make_shared<function::BasicLinearFunction>(A, u, b);

  {
    task_dynamics::ReferenceVelocity refv(ref);
    TaskDynamicsPtr tdi = refv.impl(f, constraint::Type::EQUAL, Vector2d::Zero());
    auto Value = task_dynamics::abstract::TaskDynamicsImpl::Output::Value;
    auto gl = utils::generateUpdateGraph(tdi, Value);

    gl->execute();

    FAST_CHECK_EQ(refv.order(), task_dynamics::Order::One);
    FAST_CHECK_EQ(tdi->order(), task_dynamics::Order::One);
    FAST_CHECK_UNARY(tdi->value().isApprox(ref->value()));
  }
  {
    task_dynamics::ReferenceAcceleration refa(ref);
    TaskDynamicsPtr tdi = refa.impl(f, constraint::Type::EQUAL, Vector2d::Zero());
    auto Value = task_dynamics::abstract::TaskDynamicsImpl::Output::Value;
    auto gl = utils::generateUpdateGraph(tdi, Value);

    gl->execute();

    FAST_CHECK_EQ(refa.order(), task_dynamics::Order::Two);
    FAST_CHECK_EQ(tdi->order(), task_dynamics::Order::Two);
    FAST_CHECK_UNARY(tdi->value().isApprox(ref->value()));
  }
}

TEST_CASE("Test Clamped")
{
  VariablePtr x = Space(3).createVariable("x");
  x << 2, 3, -1;
  auto f = std::make_shared<function::IdentityFunction>(x);

  double kp = 1;
  task_dynamics::Clamped<task_dynamics::P> c(1.0, kp);
  VectorXd rhs = Vector3d::Ones();

  TaskDynamicsPtr tdi = c.impl(f, constraint::Type::EQUAL, rhs);
  auto Value = task_dynamics::abstract::TaskDynamicsImpl::Output::Value;
  auto gl = utils::generateUpdateGraph(tdi, Value);
  gl->execute();

  FAST_CHECK_EQ(c.order(), task_dynamics::Order::One);
  FAST_CHECK_EQ(tdi->order(), task_dynamics::Order::One);
  FAST_CHECK_UNARY(tdi->value().isApprox(Vector3d(-.5, -1, 1))); // e'= -e = (-1,-2,2)

  x << 0.5, 1, 1.5;
  gl->execute();
  FAST_CHECK_UNARY(tdi->value().isApprox(Vector3d(.5, 0, -.5))); // e'= -e = (0.5,0,-0.5)
}

TEST_CASE("Test Clamped<FeedForward<PD>>")
{
  VariablePtr x = Space(3).createVariable("x");
  VariablePtr dx = dot(x);
  x << 1, 2, 3;
  dx << 1, 1, 1;
  auto f = std::make_shared<SphereFunction>(x, Vector3d(1, 0, -3), 2);

  auto provider = std::make_shared<FFProvider>(Eigen::VectorXd::Constant(1, 10.0));

  double kp = 2;
  double kv = 3;
  double max = 25;
  task_dynamics::Clamped<task_dynamics::FeedForwardPD> td(max, provider, &FFProvider::value, FFProvider::Output::Value,
                                                          kp, kv);
  Eigen::VectorXd rhs = Eigen::VectorXd::Constant(1, -1);
  TaskDynamicsPtr tdi = td.impl(f, constraint::Type::EQUAL, rhs);

  auto Value = task_dynamics::abstract::TaskDynamicsImpl::Output::Value;
  auto gl = utils::generateUpdateGraph(tdi, Value);
  gl->execute();

  FAST_CHECK_UNARY(provider->updated_);
  FAST_CHECK_EQ(td.order(), task_dynamics::Order::Two);
  FAST_CHECK_EQ(tdi->order(), task_dynamics::Order::Two);
  FAST_CHECK_EQ(tdi->value()[0], std::min(max, std::max(-max, -kp * (36 - rhs[0]) - kv * 16 + 10.0)));
  FAST_CHECK_UNARY(tdi->value()[0] == -max); // clamped error
  FAST_CHECK_UNARY(tdi->checkType<task_dynamics::PD::Impl>());
  FAST_CHECK_UNARY(tdi->checkType<decltype(td)::Impl>());
  FAST_REQUIRE_UNARY(tdi->checkType<task_dynamics::FeedForwardPD::Impl>()); // REQUIRE because we do a static_cast after
  FAST_CHECK_UNARY_FALSE(tdi->checkType<task_dynamics::Constant::Impl>());

  kp = 0.2;
  kv = 0.3;
  static_cast<task_dynamics::FeedForwardPD::Impl *>(tdi.get())->gains(kp, kv);
  gl->execute();
  FAST_CHECK_EQ(tdi->value()[0], APPROX_I386(std::min(max, std::max(-max, -kp * (36 - rhs[0]) - kv * 16 + 10.0))));
  FAST_CHECK_UNARY_FALSE(tdi->value()[0] == -max); // not clamped anymore
}

TEST_CASE("OneStepToZero")
{
  double dt = 0.1;
  VariablePtr x = Space(3).createVariable("x");
  VariablePtr dx = dot(x);
  x << 1, 2, 3;
  dx << 1, 1, 1;

  MatrixXd A = MatrixXd::Random(2, 3);
  VectorXd b = VectorXd::Random(2);
  auto f = std::make_shared<function::BasicLinearFunction>(A, x, b);

  {
    task_dynamics::OneStepToZero td(task_dynamics::Order::One, dt);
    TaskDynamicsPtr tdi = td.impl(f, constraint::Type::EQUAL, Vector2d::Zero());
    auto Value = task_dynamics::abstract::TaskDynamicsImpl::Output::Value;
    auto gl = utils::generateUpdateGraph(tdi, Value);

    gl->execute();

    Vector2d v = tdi->value();
    Vector3d d1x = A.colPivHouseholderQr().solve(v);
    Vector3d x1 = x->value() + dt * d1x;

    FAST_CHECK_EQ(tdi->order(), task_dynamics::Order::One);
    FAST_CHECK_UNARY((A * x1 + b).isZero(1e-8));
  }
  {
    task_dynamics::OneStepToZero td(task_dynamics::Order::Two, dt);
    TaskDynamicsPtr tdi = td.impl(f, constraint::Type::EQUAL, Vector2d::Zero());
    auto Value = task_dynamics::abstract::TaskDynamicsImpl::Output::Value;
    auto gl = utils::generateUpdateGraph(tdi, Value);

    gl->execute();

    Vector2d v = tdi->value();
    Vector3d d2x = A.colPivHouseholderQr().solve(v);
    Vector3d x1 = x->value() + dt * dx->value() + dt * dt / 2 * d2x;

    FAST_CHECK_EQ(tdi->order(), task_dynamics::Order::Two);
    FAST_CHECK_UNARY((A * x1 + b).isZero(1e-8));
  }
}
