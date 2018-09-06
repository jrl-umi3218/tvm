#include "SolverTestFunctions.h"

#include <iostream>

#include <SpaceVecAlg/SpaceVecAlg>

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/function/BasicLinearFunction.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/function/abstract/LinearFunction.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/hint/Substitution.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/VelocityDamper.h>

using namespace tvm;
using namespace Eigen;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

TEST_CASE("WrenchDistribQP")
{
  Space wrenchSpace(6);

  sva::PTransformd X_0_lc = Vector3d{0.035, -0.09, 0.0};
  sva::PTransformd X_0_rc = Vector3d{0.035, 0.09, 0.0};
  sva::PTransformd X_0_la = sva::PTransformd(Vector3d{-0.015, -0.01, 0.}) * X_0_lc;
  sva::PTransformd X_0_ra = sva::PTransformd(Vector3d{-0.015, +0.01, 0.}) * X_0_rc;

  Vector3d p_com = {0.35, 0., 0.8};
  Vector3d mg = 38. * Vector3d{0., 0., -9.81};
  sva::ForceVecd w_d = {-p_com.cross(mg), -mg};

  constexpr double MIN_PRESSURE = 15; // [N]
  constexpr double COMPLIANCE_WEIGHT = 100.;
  constexpr double NET_WRENCH_WEIGHT = 10000.;
  constexpr double PRESSURE_WEIGHT = 1.;
  constexpr double X = 0.112; // [m]
  constexpr double Y = 0.065; // [m]
  constexpr double mu = 0.7;
  MatrixXd C(16, 6);
  C << // mx,  my,  mz,  fx,  fy,            fz,
           0,   0,   0,  -1,   0,           -mu,
           0,   0,   0,  +1,   0,           -mu,
           0,   0,   0,   0,  -1,           -mu,
           0,   0,   0,   0,  +1,           -mu,
          -1,   0,   0,   0,   0,            -Y,
          +1,   0,   0,   0,   0,            -Y,
           0,  -1,   0,   0,   0,            -X,
           0,  +1,   0,   0,   0,            -X,
         +mu, +mu,  -1,  -Y,  -X, -(X + Y) * mu,
         +mu, -mu,  -1,  -Y,  +X, -(X + Y) * mu,
         -mu, +mu,  -1,  +Y,  -X, -(X + Y) * mu,
         -mu, -mu,  -1,  +Y,  +X, -(X + Y) * mu,
         +mu, +mu,  +1,  +Y,  +X, -(X + Y) * mu,
         +mu, -mu,  +1,  +Y,  -X, -(X + Y) * mu,
         -mu, +mu,  +1,  -Y,  +X, -(X + Y) * mu,
         -mu, -mu,  +1,  -Y,  -X, -(X + Y) * mu;

  Vector6d ankleWeights;
  ankleWeights << 1., 1., 1e-8, 1e-6, 1e-6, 1e-8;

  double lfr = 0.7;

  VariablePtr w_l_0 = wrenchSpace.createVariable("w_l_0");
  VariablePtr w_r_0 = wrenchSpace.createVariable("w_r_0");

  auto leftFootFric = std::make_shared<function::BasicLinearFunction>(C * X_0_lc.dualMatrix(), w_l_0);
  auto rightFootFric = std::make_shared<function::BasicLinearFunction>(C * X_0_rc.dualMatrix(), w_r_0);
  auto leftFootMinPressure = std::make_shared<function::BasicLinearFunction>(X_0_lc.dualMatrix().bottomRows<1>(), w_l_0);
  auto rightFootMinPressure = std::make_shared<function::BasicLinearFunction>(X_0_rc.dualMatrix().bottomRows<1>(), w_r_0);
  auto netWrench = std::make_shared<function::BasicLinearFunction>(
      std::vector<MatrixConstRef>{Matrix6d::Identity(), Matrix6d::Identity()},
      std::vector<VariablePtr>{w_l_0, w_r_0},
      -w_d.vector());
  auto leftAnkleWrench = std::make_shared<function::BasicLinearFunction>(X_0_la.dualMatrix(), w_l_0);
  auto rightAnkleWrench = std::make_shared<function::BasicLinearFunction>(X_0_ra.dualMatrix(), w_r_0);
  auto pressureRatio = std::make_shared<function::BasicLinearFunction>(1, std::vector<VariablePtr>{w_l_0, w_r_0});
  pressureRatio->A((1 - lfr) * X_0_lc.dualMatrix().bottomRows<1>(), *w_l_0);
  pressureRatio->A(-lfr * X_0_rc.dualMatrix().bottomRows<1>(), *w_r_0);
  //auto pressureRatio = std::make_shared<function::BasicLinearFunction>(
  //    std::vector<MatrixConstRef>{(1 - lfr) * X_0_lc.dualMatrix().bottomRows<1>(), - lfr * X_0_rc.dualMatrix().bottomRows<1>()},
  //    std::vector<VariablePtr>{w_l_0, w_r_0},
  //    Matrix<double, 1, 1>::Zero());

  LinearizedControlProblem problem;
  auto leftFootFricTask = problem.add(leftFootFric <= 0., task_dynamics::None(), { requirements::PriorityLevel(0) });
  auto rightFootFricTask = problem.add(rightFootFric <= 0., task_dynamics::None(), { requirements::PriorityLevel(0) });
  auto leftFootMinPressureTask = problem.add(leftFootMinPressure >= MIN_PRESSURE, task_dynamics::None(), { requirements::PriorityLevel(0) });
  auto rightFootMinPressureTask = problem.add(rightFootMinPressure >= MIN_PRESSURE, task_dynamics::None(), { requirements::PriorityLevel(0) });
  auto netWrenchTask = problem.add(netWrench == 0., task_dynamics::None(), { requirements::PriorityLevel(1), requirements::Weight(NET_WRENCH_WEIGHT) });
  auto leftAnkleWrenchTask = problem.add(leftAnkleWrench == 0., task_dynamics::None(), { requirements::PriorityLevel(1), requirements::AnisotropicWeight(ankleWeights), requirements::Weight(COMPLIANCE_WEIGHT) });
  auto rightAnkleWrenchTask = problem.add(rightAnkleWrench == 0., task_dynamics::None(), { requirements::PriorityLevel(1), requirements::AnisotropicWeight(ankleWeights), requirements::Weight(COMPLIANCE_WEIGHT) });
  auto pressureRatioTask = problem.add(pressureRatio == 0., task_dynamics::None(), { requirements::PriorityLevel(1), requirements::Weight(PRESSURE_WEIGHT) });

  scheme::WeightedLeastSquares solver;
  solver.solve(problem);

  Vector6d w_l_la = X_0_la.dualMatrix() * w_l_0->value();
  Vector6d w_r_ra = X_0_ra.dualMatrix() * w_r_0->value();

  lfr = 1. - lfr;
  pressureRatio->A((1 - lfr) * X_0_lc.dualMatrix().bottomRows<1>(), *w_l_0);
  pressureRatio->A(-lfr * X_0_rc.dualMatrix().bottomRows<1>(), *w_r_0);
  solver.solve(problem);

  Vector6d w_l_la2 = X_0_la.dualMatrix() * w_l_0->value();
  Vector6d w_r_ra2 = X_0_ra.dualMatrix() * w_r_0->value();
  double leftPressureDiff = (w_r_ra - w_l_la2)(5);
  double rightPressureDiff = (w_l_la - w_r_ra2)(5);
  FAST_CHECK_UNARY(std::abs(leftPressureDiff) < 1e-10);
  FAST_CHECK_UNARY(std::abs(rightPressureDiff) < 1e-10);
}
