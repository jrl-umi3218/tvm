/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#include <tvm/manifold/Real.h>
#include <tvm/manifold/SO3.h>
#include <tvm/manifold/S3.h>
#include <tvm/manifold/operators/Plus.h>
#include <tvm/manifold/operators/RMinus.h>
#include <tvm/manifold/operators/LMinus.h>

#include <Eigen/Geometry>

#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"
#include "doctestUtils.h"

using namespace tvm;
using namespace Eigen;
using namespace doctest;

template<typename BinOp, typename FromSide, typename ToSide>
void checkJacobian(const typename BinOp::repr_t& X, const typename BinOp::repr_t& Y, FromSide fs = {}, ToSide ts = {})
{
  using In = typename BinOp::group_t;
  using Out = typename BinOp::out_group_t;
  MatrixXd Jx0, Jy0;
  std::tie(Jx0, Jy0) = BinOp::jacobians(X, Y, fs, ts);
  typename BinOp::ret_t v0 = BinOp::value(X, Y);
  MatrixXd Jx, Jy;
  Jx.resizeLike(Jx0);
  Jy.resizeLike(Jy0);
  auto dim = Jx.cols();
  MatrixXd I = MatrixXd::Identity(dim, dim);
  double h = 1e-8;

  for (int i = 0; i <dim; ++i)
  {
    VectorXd hei = h * I.col(i);
    typename BinOp::ret_t vxi;
    typename BinOp::ret_t vyi;
    if (std::is_same_v<FromSide, manifold::internal::right_t>)
    {
      vxi = BinOp::value(In::compose(X, In::exp(hei)), Y);
      vyi = BinOp::value(X, In::compose(Y, In::exp(hei)));
    }
    else
    {
      vxi = BinOp::value(In::compose(In::exp(hei), X), Y);
      vyi = BinOp::value(X, In::compose(In::exp(hei), Y));
    }
    if (std::is_same_v<ToSide, manifold::internal::right_t>)
    {
      Jx.col(i) = Out::log(Out::compose(Out::inverse(v0), vxi)) / h;
      Jy.col(i) = Out::log(Out::compose(Out::inverse(v0), vyi)) / h;
    }
    else
    {
      Jx.col(i) = Out::log(Out::compose(vxi, Out::inverse(v0))) / h;
      Jy.col(i) = Out::log(Out::compose(vyi, Out::inverse(v0))) / h;
    }
  }

  FAST_CHECK_EQ(Jx0, MApprox(Jx).precision(1e-5));
  FAST_CHECK_EQ(Jy0, MApprox(Jy).precision(1e-5));
}

template<typename BinOp>
void checkJacobians(const typename BinOp::repr_t& X, const typename BinOp::repr_t& Y)
{
  using right = manifold::internal::right_t;
  using left = manifold::internal::left_t;
  checkJacobian<BinOp, right, right>(X, Y);
  checkJacobian<BinOp, right, left>(X, Y);
  checkJacobian<BinOp, left, right>(X, Y);
  checkJacobian<BinOp, left, left>(X, Y);
}

TEST_CASE("Plus")
{
  using R3 = manifold::Real<3>;
  using Rn = manifold::Real<Dynamic>;
  using SO3 = manifold::SO3;
  using S3 = manifold::S3;
  using namespace manifold::operators;
  checkJacobians<Plus<R3>>(Vector3d::Random(), Vector3d::Random());
  checkJacobians<Plus<Rn>>(VectorXd::Random(8), VectorXd::Random(8));
  checkJacobians<Plus<SO3>>(Quaterniond::UnitRandom().toRotationMatrix(), Quaterniond::UnitRandom().toRotationMatrix());
  checkJacobians<Plus<S3>>(Quaterniond::UnitRandom(), Quaterniond::UnitRandom());
}

TEST_CASE("RMinus")
{
  using R3 = manifold::Real<3>;
  using Rn = manifold::Real<Dynamic>;
  using SO3 = manifold::SO3;
  using S3 = manifold::S3;
  using namespace manifold::operators;
  checkJacobians<RMinus<R3>>(Vector3d::Random(), Vector3d::Random());
  checkJacobians<RMinus<Rn>>(VectorXd::Random(8), VectorXd::Random(8));
  checkJacobians<RMinus<SO3>>(Quaterniond::UnitRandom().toRotationMatrix(), Quaterniond::UnitRandom().toRotationMatrix());
  checkJacobians<RMinus<S3>>(Quaterniond::UnitRandom(), Quaterniond::UnitRandom());
}

TEST_CASE("LMinus")
{
  using R3 = manifold::Real<3>;
  using Rn = manifold::Real<Dynamic>;
  using SO3 = manifold::SO3;
  using S3 = manifold::S3;
  using namespace manifold::operators;
  checkJacobians<LMinus<R3>>(Vector3d::Random(), Vector3d::Random());
  checkJacobians<LMinus<Rn>>(VectorXd::Random(8), VectorXd::Random(8));
  checkJacobians<LMinus<SO3>>(Quaterniond::UnitRandom().toRotationMatrix(), Quaterniond::UnitRandom().toRotationMatrix());
  checkJacobians<LMinus<S3>>(Quaterniond::UnitRandom(), Quaterniond::UnitRandom());
}