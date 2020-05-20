/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include<tvm/Variable.h>
#include<tvm/function/BasicLinearFunction.h>
#include<tvm/utils/AffineExpr.h>
#include<tvm/utils/ProtoTask.h>
#include<tvm/internal/enums.h>

#include <iostream>

using namespace tvm;
using namespace tvm::utils;
using namespace tvm::function;
using namespace Eigen;

#define CREATE_VEC(x, vx) VectorXd x = vx;
#define CREATE_VAR(x, vx) \
  Space R##x(static_cast<int>(vx.size())); \
  VariablePtr x = R##x.createVariable(#x); \
  x << vx;

#define EVAL_EIG(res, expr) res = expr;
#define EVAL_AFF(res, expr) \
  BasicLinearFunction f(expr); \
  f.updateValue(); \
  res = f.value();

#define GENERATE(create, eval, ...) \
  PP_ID(PP_APPLY(CHOOSE_GEN_START, PP_NARG(__VA_ARGS__)) \
          (create, eval, __VA_ARGS__))

#define CHOOSE_GEN_START(count) GENERATE##count

#define GENERATE2(create, eval, res, expr) eval(res, expr)
#define GENERATE4(create, eval, res, expr, x, vx, ...) create(x,vx) PP_ID(GENERATE2(create, eval, res, expr))
#define GENERATE6(create, eval, res, expr, x, vx, ...) create(x,vx) PP_ID(GENERATE4(create, eval, res, expr, __VA_ARGS__))
#define GENERATE8(create, eval, res, expr, x, vx, ...) create(x,vx) PP_ID(GENERATE6(create, eval, res, expr, __VA_ARGS__))
#define GENERATE10(create, eval, res, expr, x, vx, ...) create(x,vx) PP_ID(GENERATE8(create, eval, res, expr, __VA_ARGS__))
#define GENERATE12(create, eval, res, expr, x, vx, ...) create(x,vx) PP_ID(GENERATE10(create, eval, res, expr, __VA_ARGS__))


// Called as TEST_AFFINE_EXPR(expr x1, v1, x2, v2, ...)
// Test if the expression expr (e.g. A1*x1+A2*x2+b) evaluated at xi=vi gives the
// same result as the BasicLinearFunction derived from the AffineExpr described
// by expr where the xi are VariablePtr with value vi.
// This tests in passing that the corresponding AffineExpr creation compiles
// correctly.
#define TEST_AFFINE_EXPR(expr, ...) \
{ \
  VectorXd res1, res2; \
  { \
    PP_ID(GENERATE(CREATE_VEC, EVAL_EIG, res1, expr, __VA_ARGS__)) \
  } \
  { \
    PP_ID(GENERATE(CREATE_VAR, EVAL_AFF, res2, expr, __VA_ARGS__)) \
  } \
  FAST_CHECK_UNARY((res1-res2).isZero()); \
}


TEST_CASE("Test AffineExpr compilation and validity")
{
  VectorXd vx = VectorXd::Random(5);
  VectorXd vy = VectorXd::Random(5);
  VectorXd vz = VectorXd::Random(3);
  VectorXd vw = VectorXd::Random(3);
  VectorXd vu = VectorXd::Random(6);
  MatrixXd A = MatrixXd::Random(6, 5);
  MatrixXd B = MatrixXd::Random(6, 5);
  MatrixXd C = MatrixXd::Random(6, 3);
  MatrixXd M = MatrixXd::Random(6, 6);
  VectorXd d = VectorXd::Random(6);
  VectorXd e = VectorXd::Random(6);
  
  
  TEST_AFFINE_EXPR(A * x,                                     x, vx)                      //LinearExpr, simple case
  TEST_AFFINE_EXPR(M * A * x,                                 x, vx)                      //LinearExpr, simple matrix expression
  TEST_AFFINE_EXPR((d.asDiagonal() * A + M * B) * x,          x, vx)                      // complex matrix expression
  TEST_AFFINE_EXPR(A * x + d,                                 x, vx)                      // AffineExpr = LinearExpr + vector
  TEST_AFFINE_EXPR(d + A * x,                                 x, vx)                      // AffineExpr = vector + LinearExpr
  TEST_AFFINE_EXPR(A * x + (3*d+M*e),                         x, vx)                      // AffineExpr = LinearExpr + vector(complex expr)
  TEST_AFFINE_EXPR(A * x + B * y,                             x, vx, y, vy)               // AffineExpr = LinearExpr + LinearExpr
  TEST_AFFINE_EXPR(A * x + B * y + C * z,                     x, vx, y, vy, z, vz)        // AffineExpr = AffineExpr + LinearExpr
  TEST_AFFINE_EXPR(A * x + (B * y + C * z),                   x, vx, y, vy, z, vz)        // AffineExpr = LinearExpr + AffineExpr
  TEST_AFFINE_EXPR(A * x + B * y + d,                         x, vx, y, vy)               // AffineExpr = AffineExpr(NoConstant) + vector
  TEST_AFFINE_EXPR(d + (A * x + B * y),                       x, vx, y, vy)               // AffineExpr = vector + AffineExpr(NoConstant)
  TEST_AFFINE_EXPR(A * x + d + e,                             x, vx)                      // AffineExpr = AffineExpr(with constant) + vector
  TEST_AFFINE_EXPR(A * x + d + B * y + e,                     x, vx, y, vy)               // AffineExpr = AffineExpr(with constant) + vector
  TEST_AFFINE_EXPR(e + (A * x + d),                           x, vx)                      // AffineExpr = vector + AffineExpr(with constant)
  TEST_AFFINE_EXPR(e + (A * x + d + B * y),                   x, vx, y, vy)               // AffineExpr = vector + AffineExpr(with constant)
  TEST_AFFINE_EXPR((A * x + B * y) + (C * z + C * w),         x, vx, y, vy, z, vz, w, vw) // AffineExpr = AffineExpr(NoConstant) + AffineExpr(NoConstant)
  TEST_AFFINE_EXPR((A * x + B * y) + (C * z + C * w + e),     x, vx, y, vy, z, vz, w, vw) // AffineExpr = AffineExpr(NoConstant) + AffineExpr(with constant)
  TEST_AFFINE_EXPR((A * x + B * y + d) + (C * z + C * w),     x, vx, y, vy, z, vz, w, vw) // AffineExpr = AffineExpr(with constant) + AffineExpr(with constant)
  TEST_AFFINE_EXPR((A * x + B * y + d) + (C * z + C * w + e), x, vx, y, vy, z, vz, w, vw) // AffineExpr = AffineExpr(with complex constant) + AffineExpr(with complex constant)
  TEST_AFFINE_EXPR(A * x + (C * z + B * x),                   x, vx, z, vz)               // AffineExpr = LinearExpr + AffineExpr (with duplicated variable)

  TEST_AFFINE_EXPR(A * x + u,                                 x, vx, u, vu)
  TEST_AFFINE_EXPR(u + A * x,                                 x, vx, u, vu)
  TEST_AFFINE_EXPR(x + y,                                     x, vx, y, vy)
  TEST_AFFINE_EXPR(-x,                                        x, vx)
  TEST_AFFINE_EXPR(3 * x,                                     x, vx)
  TEST_AFFINE_EXPR(u + d,                                     u, vu)
  TEST_AFFINE_EXPR(d + u,                                     u, vu)
  TEST_AFFINE_EXPR(A * x + d + u,                             x, vx, u, vu)
  TEST_AFFINE_EXPR(u + (A * x + d),                           x, vx, u, vu)
  TEST_AFFINE_EXPR(-(A*x),                                    x, vx)
  TEST_AFFINE_EXPR(-((d.asDiagonal() * A + M * B) * x),       x, vx)
  TEST_AFFINE_EXPR(3 * (A*x),                                 x, vx)
  TEST_AFFINE_EXPR(3 * M * (A*x),                             x, vx)
  TEST_AFFINE_EXPR(3 * (M *(A*x)),                            x, vx)
  TEST_AFFINE_EXPR(M *((d.asDiagonal() * A + M * B) * x),     x, vx)
  TEST_AFFINE_EXPR(-M *((d.asDiagonal() * A + M * B) * x),    x, vx)
  TEST_AFFINE_EXPR(A * x - u,                                 x, vx, u, vu)
  TEST_AFFINE_EXPR(u - A * x,                                 x, vx, u, vu)
  TEST_AFFINE_EXPR(x - y,                                     x, vx, y, vy)
  TEST_AFFINE_EXPR(u - d,                                     u, vu)
  TEST_AFFINE_EXPR(d - u,                                     u, vu)
  TEST_AFFINE_EXPR(A * x + d - u,                             x, vx, u, vu)
  TEST_AFFINE_EXPR(u - (A * x + d),                           x, vx, u, vu)
  TEST_AFFINE_EXPR(A * x - B * y,                             x, vx, y, vy)
  TEST_AFFINE_EXPR((A * x + B * y) - (C * z + C * w),         x, vx, y, vy, z, vz, w, vw)
  TEST_AFFINE_EXPR((A * x + B * y + d) - (C * z + C * w + e), x, vx, y, vy, z, vz, w, vw)
  TEST_AFFINE_EXPR(2 * (A * x + u),                           x, vx, u, vu)
  TEST_AFFINE_EXPR(M * (A * x + u),                           x, vx, u, vu)
  TEST_AFFINE_EXPR(2 * (A * x + u + d),                       x, vx, u, vu)
  TEST_AFFINE_EXPR(M * (A * x + u + d),                       x, vx, u, vu)
}

TEST_CASE("Test integration with ProtoTask")
{
  MatrixXd A = MatrixXd::Random(6, 5);
  VectorXd b = VectorXd::Random(6);
  
  VariablePtr x = Space(5).createVariable("x");

  auto p1 = (A * x == 0);
  auto p2 = (0 == A * x);
  auto p3 = (A * x >= 0);
  auto p4 = (0 >= A * x);
  auto p5 = (A * x <= 0);
  auto p6 = (0 <= A * x);

  auto p11 = (A * x + b == 0);
  auto p12 = (0 == A * x + b);
  auto p13 = (A * x + b >= 0);
  auto p14 = (0 >= A * x + b);
  auto p15 = (A * x + b <= 0);
  auto p16 = (0 <= A * x + b);
}