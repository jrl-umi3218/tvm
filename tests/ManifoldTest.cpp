/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/manifold/Real.h>
#include <tvm/manifold/SO3.h>

#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>

#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm;
using namespace Eigen;


TEST_CASE("Rn")
{
  using R3 = manifold::Real<3>;
  Vector3d x = Vector3d::Random();
  Vector3d y = Vector3d::Random();
  MatrixXd M = MatrixXd::Random(3, 4);
  auto z = M.row(0).transpose().head(3);
  FAST_CHECK_UNARY((x + y).isApprox(R3::compose(x, y)));
  FAST_CHECK_UNARY(x.isApprox(R3::compose(x, R3::identity)));
  FAST_CHECK_UNARY(y.isApprox(R3::compose(R3::identity, y)));
  FAST_CHECK_UNARY((z + x + y).isApprox(R3::compose(z, x+y)));
  FAST_CHECK_UNARY((-x).isApprox(R3::inverse(x)));
  FAST_CHECK_UNARY(std::is_same_v<decltype(R3::inverse(R3::identity)),R3::Identity>);

  auto l = R3::log(x);
  FAST_CHECK_UNARY(x.isApprox(l));
  FAST_CHECK_UNARY(std::is_same_v<decltype(R3::log(R3::identity)), R3::AlgIdentity>);
  FAST_CHECK_UNARY(x.isApprox(R3::exp(l)));
  FAST_CHECK_UNARY(std::is_same_v<decltype(R3::exp(R3::algIdentity)), R3::Identity>);
  FAST_CHECK_UNARY((x - y).isApprox(R3::compose(R3::exp(l), R3::inverse(y))));
}

TEST_CASE("SO3")
{
  using SO3 = manifold::SO3;
  Matrix3d x = Quaterniond::UnitRandom().toRotationMatrix();
  Matrix3d y = Quaterniond::UnitRandom().toRotationMatrix();
  MatrixXd M = MatrixXd::Random(3, 4);
  auto z = M.leftCols<3>();
  FAST_CHECK_UNARY((x * y).isApprox(SO3::compose(x, y)));
  FAST_CHECK_UNARY(x.isApprox(SO3::compose(x, SO3::identity)));
  FAST_CHECK_UNARY(y.isApprox(SO3::compose(SO3::identity, y)));
  FAST_CHECK_UNARY((z * x * y).isApprox(SO3::compose(z, x * y)));
  FAST_CHECK_UNARY(x.transpose().isApprox(SO3::inverse(x)));
  FAST_CHECK_UNARY(std::is_same_v<decltype(SO3::inverse(SO3::identity)), SO3::Identity>);

  auto l = SO3::log(x);
  FAST_CHECK_UNARY(SO3::vee(SO3::hat(l)).isApprox(l));
  FAST_CHECK_UNARY(Matrix3d(x.log()).isApprox(SO3::hat(l)));
  FAST_CHECK_UNARY(std::is_same_v<decltype(SO3::log(SO3::identity)), SO3::AlgIdentity>);
  FAST_CHECK_UNARY(x.isApprox(SO3::exp(l)));
  FAST_CHECK_UNARY(std::is_same_v<decltype(SO3::exp(SO3::algIdentity)), SO3::Identity>);
  FAST_CHECK_UNARY((x * y.transpose()).isApprox(SO3::compose(SO3::exp(l), SO3::inverse(y))));
}