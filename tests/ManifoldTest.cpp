/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/manifold/Real.h>
#include <tvm/manifold/SO3.h>
#include <tvm/manifold/internal/Adjoint.h>

#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>

#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm;
using namespace Eigen;


TEST_CASE("R3")
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

TEST_CASE("Rn")
{
  using Rn = manifold::Real<Dynamic>;
  VectorXd x = VectorXd::Random(8);
  VectorXd y = VectorXd::Random(8);
  MatrixXd M = MatrixXd::Random(3, 10);
  auto z = M.row(0).transpose().head(8);
  FAST_CHECK_UNARY((x + y).isApprox(Rn::compose(x, y)));
  FAST_CHECK_UNARY(x.isApprox(Rn::compose(x, Rn::identity)));
  FAST_CHECK_UNARY(y.isApprox(Rn::compose(Rn::identity, y)));
  FAST_CHECK_UNARY((z + x + y).isApprox(Rn::compose(z, x + y)));
  FAST_CHECK_UNARY((-x).isApprox(Rn::inverse(x)));
  FAST_CHECK_UNARY(std::is_same_v<decltype(Rn::inverse(Rn::identity)), Rn::Identity>);

  auto l = Rn::log(x);
  FAST_CHECK_UNARY(x.isApprox(l));
  FAST_CHECK_UNARY(std::is_same_v<decltype(Rn::log(Rn::identity)), Rn::AlgIdentity>);
  FAST_CHECK_UNARY(x.isApprox(Rn::exp(l)));
  FAST_CHECK_UNARY(std::is_same_v<decltype(Rn::exp(Rn::algIdentity)), Rn::Identity>);
  FAST_CHECK_UNARY((x - y).isApprox(Rn::compose(Rn::exp(l), Rn::inverse(y))));
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

template<typename LG, typename Expr>
struct checkAdjoint
{
  template<typename Adj>
  static void run(const Adj& adj, bool inverse, bool transpose, bool sign,
    const typename LG::repr_t& opValue, const typename Adj::matrix_t& mat)
  {
    FAST_CHECK_UNARY(std::is_same_v<typename Adj::LG, LG>);
    FAST_CHECK_UNARY(std::is_same_v<typename Adj::Expr, Expr>);
    FAST_CHECK_EQ(Adj::Inverse, inverse);
    FAST_CHECK_EQ(Adj::Transpose, transpose);
    FAST_CHECK_EQ(Adj::PositiveSign, sign);
    if constexpr (!std::is_same_v<Expr,typename LG::Identity>)
      FAST_CHECK_UNARY(opValue.isApprox(adj.operand()));
    if constexpr (LG::dim>=0 || !Adj::template isId<Adj::Expr>::value)
      FAST_CHECK_UNARY(mat.isApprox(adj.matrix()));
    else
      FAST_CHECK_UNARY(mat.isApprox(adj.matrix(opValue.size())));
  }
};

TEST_CASE("AdjointR3")
{
  using R3 = manifold::Real<3>;
  using tvm::manifold::internal::ReprOwn;
  Vector3d x = Vector3d::Random();
  Vector3d y = Vector3d::Random();
  tvm::manifold::internal::Adjoint<R3> AdX(x);
  tvm::manifold::internal::Adjoint<R3> AdY(y);
  tvm::manifold::internal::Adjoint AdI(R3::Identity{});
  auto AdXY = AdX * AdY;
  auto AdXI = AdX * AdI;
  auto AdIX = AdI * AdX;
  auto AdII = AdI * AdI;
  checkAdjoint<R3, ReprOwn<R3>>::run(AdXY, false, false, true, x + y, Matrix3d::Identity());
  checkAdjoint<R3, ReprOwn<R3>>::run(AdXI, false, false, true, x, Matrix3d::Identity());
  checkAdjoint<R3, ReprOwn<R3>>::run(AdIX, false, false, true, x, Matrix3d::Identity());
  checkAdjoint<R3, decltype(AdI)::Expr>::run(AdII, false, false, true, {}, Matrix3d::Identity());
}

TEST_CASE("AdjointRn")
{
  using Rn = manifold::Real<Dynamic>;
  using tvm::manifold::internal::ReprOwn;
  VectorXd x = VectorXd::Random(8);
  VectorXd y = VectorXd::Random(8);
  tvm::manifold::internal::Adjoint<Rn> AdX(x);
  tvm::manifold::internal::Adjoint<Rn> AdY(y);
  tvm::manifold::internal::Adjoint AdI(Rn::Identity{});
  auto AdXY = AdX * AdY;
  auto AdXI = AdX * AdI;
  auto AdIX = AdI * AdX;
  auto AdII = AdI * AdI;
  checkAdjoint<Rn, ReprOwn<Rn>>::run(AdXY, false, false, true, x + y, MatrixXd::Identity(8, 8));
  checkAdjoint<Rn, ReprOwn<Rn>>::run(AdXI, false, false, true, x, MatrixXd::Identity(8, 8));
  checkAdjoint<Rn, ReprOwn<Rn>>::run(AdIX, false, false, true, x, MatrixXd::Identity(8, 8));
  checkAdjoint<Rn, decltype(AdI)::Expr>::run(AdII, false, false, true, VectorXd::Zero(8), MatrixXd::Identity(8, 8));
}

TEST_CASE("AdjointSO3")
{
  using SO3 = manifold::SO3;
  using tvm::manifold::internal::ReprOwn;
  Matrix3d x = Quaterniond::UnitRandom().toRotationMatrix();
  Matrix3d y = Quaterniond::UnitRandom().toRotationMatrix();
  tvm::manifold::internal::Adjoint<SO3> AdX(x);
  tvm::manifold::internal::Adjoint<SO3> AdY(y);
  tvm::manifold::internal::Adjoint AdI(SO3::Identity{});
  auto AdXY = AdX * AdY;
  auto AdXI = AdX * AdI;
  auto AdIX = AdI * AdX;
  auto AdII = AdI * AdI;
  checkAdjoint<SO3, ReprOwn<SO3>>::run(AdXY, false, false, true, x * y, x * y);
  checkAdjoint<SO3, ReprOwn<SO3>>::run(AdXI, false, false, true, x, x);
  checkAdjoint<SO3, ReprOwn<SO3>>::run(AdIX, false, false, true, x, x);
  checkAdjoint<SO3, decltype(AdI)::Expr>::run(AdII, false, false, true, {}, Matrix3d::Identity());
}