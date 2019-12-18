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
using namespace tvm::requirements;
using namespace Eigen;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

/** Verbosity setting of this test.
 *
 */
constexpr bool VERBOSE = false;

/** Minimum pressure under each contact.
 *
 */
constexpr double MIN_PRESSURE = 15; // [N]

/** Problem weights and their square roots.
 *
 */
const double COMPLIANCE_WEIGHT = 100.;
const double COMPLIANCE_WEIGHT_SQRT = std::sqrt(COMPLIANCE_WEIGHT);
const double NET_WRENCH_WEIGHT = 10000.;
const double NET_WRENCH_WEIGHT_SQRT = std::sqrt(NET_WRENCH_WEIGHT);
const double PRESSURE_WEIGHT = 1.;
const double PRESSURE_WEIGHT_SQRT = std::sqrt(PRESSURE_WEIGHT);

/** Robot state for wrench distribution.
 *
 */
struct RobotState
{
  /** Default robot state.
   *
   */
  RobotState()
    : wrenchFaceMatrix(16, 6)
  {
    X_0_lc = Vector3d{0.035, -0.09, 0.0};
    X_0_rc = Vector3d{0.035, 0.09, 0.0};
    X_0_la = sva::PTransformd(Vector3d{-0.015, -0.01, 0.}) * X_0_lc;
    X_0_ra = sva::PTransformd(Vector3d{-0.015, +0.01, 0.}) * X_0_rc;

    Vector3d p_com = {0.35, 0., 0.8};
    Vector3d mg = 38. * Vector3d{0., 0., -9.81};
    w_d = {-p_com.cross(mg), -mg};

    ankleWeights << 1., 1., 1e-8, 1e-6, 1e-6, 1e-8;
    lfr = 0.7;

    constexpr double X = 0.112; // [m]
    constexpr double Y = 0.065; // [m]
    constexpr double mu = 0.7;
    wrenchFaceMatrix << 
      // mx,  my,  mz,  fx,  fy,            fz,
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
  }

public:
  Eigen::MatrixXd wrenchFaceMatrix; /**< Inequality matrix of contact wrench cone */
  Eigen::Vector6d ankleWeights; /**< Ankle anisotropic weights */
  double lfr; /**< Left foot pressure ratio */
  sva::ForceVecd w_d; /**< Desired net wrench */
  sva::PTransformd X_0_la; /**< Plucker transform to left ankle */
  sva::PTransformd X_0_lc; /**< Plucker transform to left foot center */
  sva::PTransformd X_0_ra; /**< Plucker transform to right ankle */
  sva::PTransformd X_0_rc; /**< Plucker transform to right foot center */
};

/** Wrench distribution QP with matrices built "by hand".
 *
 * \param robot Robot state, provided by test case.
 *
 * \returns x Solution vector concatenating [w_l_0; w_r_0].
 *
 */
Eigen::VectorXd distributeWrenchGroundTruth(const RobotState & robot)
{
  const Eigen::MatrixXd & wrenchFaceMatrix = robot.wrenchFaceMatrix;
  const Eigen::Vector6d & ankleWeights = robot.ankleWeights;
  const sva::ForceVecd & w_d = robot.w_d;
  const sva::PTransformd & X_0_la = robot.X_0_la;
  const sva::PTransformd & X_0_lc = robot.X_0_lc;
  const sva::PTransformd & X_0_ra = robot.X_0_ra;
  const sva::PTransformd & X_0_rc = robot.X_0_rc;
  double lfr = robot.lfr;

  // Variables
  // ---------
  // x = [w_l_0 w_r_0] where
  // w_l_0: spatial force vector of left foot contact in inertial frame
  // w_r_0: spatial force vector of right foot contact in inertial frame
  //
  // Objective
  // ---------
  // Weighted minimization of the following tasks:
  // w_l_0 + w_r_0 == w_d -- realize desired contact wrench
  // w_l_la == 0 -- minimize left foot ankle torque (anisotropic weight)
  // w_r_ra == 0 -- minimize right foot ankle torque (anisotropic weight)
  // (1 - lfr) * w_l_lc.z() == lfr * w_r_rc.z()
  //
  // Constraints
  // -----------
  // CWC X_0_lc* w_l_0 <= 0  -- left foot wrench within contact wrench cone
  // CWC X_0_rc* w_r_0 <= 0  -- right foot wrench within contact wrench cone
  // (X_0_lc* w_l_0).z() > minPressure  -- minimum left foot contact pressure
  // (X_0_rc* w_r_0).z() > minPressure  -- minimum right foot contact pressure

  constexpr unsigned NB_VAR = 6 + 6;
  constexpr unsigned COST_DIM = 6 + NB_VAR + 1;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  A.setZero(COST_DIM, NB_VAR);
  b.setZero(COST_DIM);

  // |w_l_0 + w_r_0 - w_d|^2
  auto A_net = A.block<6, 12>(0, 0);
  auto b_net = b.segment<6>(0);
  A_net.block<6, 6>(0, 0) = Eigen::Matrix6d::Identity();
  A_net.block<6, 6>(0, 6) = Eigen::Matrix6d::Identity();
  b_net = w_d.vector();

  // |ankle torques|^2
  auto A_la = A.block<6, 6>(6, 0);
  auto A_ra = A.block<6, 6>(12, 6);
  // anisotropic weights:  taux, tauy, tauz,   fx,   fy,   fz;
  A_la.diagonal() = ankleWeights.cwiseSqrt();
  A_ra.diagonal() = ankleWeights.cwiseSqrt();
  A_la *= X_0_la.dualMatrix();
  A_ra *= X_0_ra.dualMatrix();

  // |(1 - lfr) * w_l_lc.force().z() - lfr * w_r_rc.force().z()|^2
  auto A_pressure = A.block<1, 12>(18, 0);
  A_pressure.block<1, 6>(0, 0) = (1 - lfr) * X_0_lc.dualMatrix().bottomRows<1>();
  A_pressure.block<1, 6>(0, 6) = -lfr * X_0_rc.dualMatrix().bottomRows<1>();

  // Apply weights (b_la = b_ra = b_pressure = 0)
  A_net *= NET_WRENCH_WEIGHT_SQRT;
  b_net *= NET_WRENCH_WEIGHT_SQRT;
  A_la *= COMPLIANCE_WEIGHT_SQRT;
  A_ra *= COMPLIANCE_WEIGHT_SQRT;
  A_pressure *= PRESSURE_WEIGHT_SQRT;

  constexpr unsigned CONS_DIM = 16 + 16 + 2;
  Eigen::Matrix<double, CONS_DIM , NB_VAR> C;
  Eigen::VectorXd bl, bu;
  C.setZero(CONS_DIM, NB_VAR);
  bl.setConstant(NB_VAR + CONS_DIM, -1e5);
  bu.setConstant(NB_VAR + CONS_DIM, +1e5);
  auto blCons = bl.tail<CONS_DIM>();
  auto buCons = bu.tail<CONS_DIM>();
  // CWC * w_l_lc <= 0
  C.block<16, 6>(0, 0) = wrenchFaceMatrix * X_0_lc.dualMatrix();
  buCons.segment<16>(0).setZero();
  // CWC * w_r_rc <= 0
  C.block<16, 6>(16, 6) = wrenchFaceMatrix * X_0_rc.dualMatrix();
  buCons.segment<16>(16).setZero();
  // w_l_lc.force().z() >= MIN_PRESSURE
  C.block<1, 6>(32, 0) = X_0_lc.dualMatrix().bottomRows<1>();
  // w_r_rc.force().z() >= MIN_PRESSURE
  C.block<1, 6>(33, 6) = X_0_rc.dualMatrix().bottomRows<1>();
  blCons.segment<2>(32).setConstant(MIN_PRESSURE);
  buCons.segment<2>(32).setConstant(+1e5);

  if (VERBOSE)
  {
    std::cout << "A_gt =\n" << A << std::endl;
    std::cout << "b_gt =\t" << b.transpose() << std::endl;
    std::cout << "C_gt =\n" << C << std::endl;
    std::cout << "l_gt =\t" << bl.transpose() << std::endl;
    std::cout << "u_gt =\t" << bu.transpose() << std::endl;
  }

  Eigen::LSSOL_LS solver;
  solver.solve(A, b, C, bl, bu);
  Eigen::VectorXd x = solver.result();
  if (solver.inform() != Eigen::lssol::eStatus::STRONG_MINIMUM)
  {
    std::cout << "Wrench distribution QP failed to run" << std::endl;
    solver.print_inform();
  }
  return x;
}

/** Check that solutions found by TVM and hand-made QP are the same.
 *
 * \param robot Robot state, provided by test case.
 *
 * \param w_l_0 Left foot wrench from TVM solution.
 *
 * \param w_r_0 Right foot wrench from TVM solution.
 *
 */
bool checkSolution(const RobotState & robot, Eigen::Vector6d w_l_0, Eigen::Vector6d w_r_0)
{
  Eigen::VectorXd x = distributeWrenchGroundTruth(robot);
  Eigen::Vector6d w_l_0_gt = x.segment<6>(0);
  Eigen::Vector6d w_r_0_gt = x.segment<6>(6);

  if (VERBOSE)
  {
    std::cout << "w_l_0 = " << w_l_0.transpose() << std::endl;
    std::cout << "w_r_0 = " << w_r_0.transpose() << std::endl;
    std::cout << "w_l_0_gt = " << w_l_0_gt.transpose() << std::endl;
    std::cout << "w_r_0_gt = " << w_r_0_gt.transpose() << std::endl;
  }

  constexpr double EPSILON = 1e-3;
  return ((w_l_0 - w_l_0_gt).norm() < EPSILON && (w_r_0 - w_r_0_gt).norm() < EPSILON);
}

TEST_CASE("WrenchDistribQP")
{
  RobotState robot;
  const Eigen::MatrixXd & wrenchFaceMatrix = robot.wrenchFaceMatrix;
  const Eigen::Vector6d & ankleWeights = robot.ankleWeights;
  const sva::ForceVecd & w_d = robot.w_d;
  const sva::PTransformd & X_0_la = robot.X_0_la;
  const sva::PTransformd & X_0_lc = robot.X_0_lc;
  const sva::PTransformd & X_0_ra = robot.X_0_ra;
  const sva::PTransformd & X_0_rc = robot.X_0_rc;

  Space wrenchSpace(6);
  VariablePtr w_l_0 = wrenchSpace.createVariable("w_l_0");
  VariablePtr w_r_0 = wrenchSpace.createVariable("w_r_0");

  LinearizedControlProblem problem;
  auto leftFootFricTask         = problem.add(wrenchFaceMatrix * X_0_lc.dualMatrix() * w_l_0 <= 0.                        , { PriorityLevel(0) });
  auto rightFootFricTask        = problem.add(wrenchFaceMatrix * X_0_rc.dualMatrix() * w_r_0 <= 0.                        , { PriorityLevel(0) });
  auto leftFootMinPressureTask  = problem.add(X_0_lc.dualMatrix().bottomRows<1>() * w_l_0 >= MIN_PRESSURE                 , { PriorityLevel(0) });
  auto rightFootMinPressureTask = problem.add(X_0_rc.dualMatrix().bottomRows<1>() * w_r_0 >= MIN_PRESSURE                 , { PriorityLevel(0) });
  auto netWrenchTask            = problem.add(Matrix6d::Identity() * w_l_0 + Matrix6d::Identity() * w_r_0 == w_d.vector() , { PriorityLevel(1), Weight(NET_WRENCH_WEIGHT) });
  auto leftAnkleWrenchTask      = problem.add(X_0_la.dualMatrix() * w_l_0 == 0.                                           , { PriorityLevel(1), AnisotropicWeight(ankleWeights), Weight(COMPLIANCE_WEIGHT) });
  auto rightAnkleWrenchTask     = problem.add(X_0_ra.dualMatrix() * w_r_0 == 0.                                           , { PriorityLevel(1), AnisotropicWeight(ankleWeights), Weight(COMPLIANCE_WEIGHT) });
  auto pressureRatioTask        = problem.add((1 - robot.lfr) * X_0_lc.dualMatrix().bottomRows<1>() * w_l_0 
                                            + (-robot.lfr) * X_0_rc.dualMatrix().bottomRows<1>() * w_r_0 == 0             , { PriorityLevel(1), Weight(PRESSURE_WEIGHT) });

  // First problem with initial left foot ratio
  scheme::WeightedLeastSquares solver(VERBOSE);
  solver.solve(problem);
  FAST_CHECK_UNARY(checkSolution(robot, w_l_0->value(), w_r_0->value()));
  Vector6d w_l_la1 = X_0_la.dualMatrix() * w_l_0->value();
  Vector6d w_r_ra1 = X_0_ra.dualMatrix() * w_r_0->value();

  // Second problem with complementary left foot ratio
  auto pressureRatio = std::static_pointer_cast<function::BasicLinearFunction>(pressureRatioTask->task.function());
  robot.lfr = 1. - robot.lfr;
  pressureRatio->A((1 - robot.lfr) * X_0_lc.dualMatrix().bottomRows<1>(), *w_l_0);
  pressureRatio->A(-robot.lfr * X_0_rc.dualMatrix().bottomRows<1>(), *w_r_0);
  solver.solve(problem);
  FAST_CHECK_UNARY(checkSolution(robot, w_l_0->value(), w_r_0->value()));
  Vector6d w_l_la2 = X_0_la.dualMatrix() * w_l_0->value();
  Vector6d w_r_ra2 = X_0_ra.dualMatrix() * w_r_0->value();

  // Check that the two solutions are symmetric
  double leftPressureDiff = (w_r_ra1 - w_l_la2)(5);
  double rightPressureDiff = (w_l_la1 - w_r_ra2)(5);
  FAST_CHECK_UNARY(std::abs(leftPressureDiff) < 1e-10);
  FAST_CHECK_UNARY(std::abs(rightPressureDiff) < 1e-10);
}
