/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Robot.h>
#include <tvm/Task.h>
#include <tvm/robot/CoMFunction.h>
#include <tvm/robot/CoMInConvexFunction.h>
#include <tvm/robot/CollisionFunction.h>
#include <tvm/robot/ConvexHull.h>
#include <tvm/robot/JointsSelector.h>
#include <tvm/robot/OrientationFunction.h>
#include <tvm/robot/PositionFunction.h>
#include <tvm/robot/PostureFunction.h>
#include <tvm/robot/internal/DynamicFunction.h>
#include <tvm/robot/internal/GeometricContactFunction.h>
#include <tvm/robot/utils.h>

#include <tvm/Clock.h>
#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/hint/Substitution.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/solver/QuadprogLeastSquareSolver.h>
#include <tvm/solver/defaultLeastSquareSolver.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/VelocityDamper.h>
#include <tvm/utils/sch.h>

#include <RBDyn/parsers/urdf.h>

#include <RBDyn/ID.h>

#include <fstream>
#include <iostream>

#include "RobotPublisher.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

static std::string jvrc_path = JVRC_DESCRIPTION_PATH;
static std::string env_path = MC_ENV_DESCRIPTION_PATH;
static std::string jvrc_urdf = jvrc_path + "/urdf/jvrc1.urdf";
static std::string ground_urdf = env_path + "/urdf/ground.urdf";
static std::string jvrc_convex_path = jvrc_path + "/convex/jvrc1/";

tvm::robot::ConvexHullPtr loadConvex(tvm::robot::FramePtr f)
{
  return std::make_shared<tvm::robot::ConvexHull>(jvrc_convex_path + f->body() + "-ch.txt", f,
                                                  sva::PTransformd::Identity());
}

/** Build a cube as a set of planes from a given origin and size */
std::vector<tvm::geometry::PlanePtr> makeCube(const Eigen::Vector3d & origin, double size)
{
  return {std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{1, 0, 0}, origin + Eigen::Vector3d{-size, 0, 0}),
          std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{-1, 0, 0}, origin + Eigen::Vector3d{size, 0, 0}),
          std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{0, 1, 0}, origin + Eigen::Vector3d{0, -size, 0}),
          std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{0, -1, 0}, origin + Eigen::Vector3d{0, size, 0}),
          std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{0, 0, 1}, origin + Eigen::Vector3d{0, 0, -size}),
          std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{0, 0, -1}, origin + Eigen::Vector3d{0, 0, size})};
}

#if defined(TVM_USE_LSSOL) || defined(TVM_USE_QLD) // Quadprog is having trouble, seemingly with the bounds on f
TEST_CASE("Test a problem with a robot")
{
#  if NDEBUG
  size_t iter = 2000;
#  else
  size_t iter = 100;
#  endif
  double dt = 0.005;

  tvm::ControlProblem pb;
  tvm::Clock clock(dt);

  std::map<std::string, std::vector<double>> ref_q = {{"Root", {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8275}},
                                                      {"R_HIP_P", {-0.38}},
                                                      {"R_HIP_R", {-0.01}},
                                                      {"R_HIP_Y", {0.0}},
                                                      {"R_KNEE", {0.72}},
                                                      {"R_ANKLE_R", {-0.01}},
                                                      {"R_ANKLE_P", {-0.33}},
                                                      {"L_HIP_P", {-0.38}},
                                                      {"L_HIP_R", {0.02}},
                                                      {"L_HIP_Y", {0.0}},
                                                      {"L_KNEE", {0.72}},
                                                      {"L_ANKLE_R", {-0.02}},
                                                      {"L_ANKLE_P", {-0.33}},
                                                      {"WAIST_Y", {0.0}},
                                                      {"WAIST_P", {0.13}},
                                                      {"WAIST_R", {0.00}},
                                                      {"NECK_Y", {0.0}},
                                                      {"NECK_R", {0.0}},
                                                      {"NECK_P", {0.0}},
                                                      {"R_SHOULDER_P", {-0.052}},
                                                      {"R_SHOULDER_R", {-0.17}},
                                                      {"R_SHOULDER_Y", {0.0}},
                                                      {"R_ELBOW_P", {-0.52}},
                                                      {"R_ELBOW_Y", {0.0}},
                                                      {"R_WRIST_R", {0.0}},
                                                      {"R_WRIST_Y", {0.0}},
                                                      {"R_UTHUMB", {0.0}},
                                                      {"R_LTHUMB", {0.0}},
                                                      {"R_UINDEX", {0.0}},
                                                      {"R_LINDEX", {0.0}},
                                                      {"R_ULITTLE", {0.0}},
                                                      {"R_LLITTLE", {0.0}},
                                                      {"L_SHOULDER_P", {-0.052000}},
                                                      {"L_SHOULDER_R", {0.17}},
                                                      {"L_SHOULDER_Y", {0.0}},
                                                      {"L_ELBOW_P", {-0.52}},
                                                      {"L_ELBOW_Y", {0.0}},
                                                      {"L_WRIST_R", {0.0}},
                                                      {"L_WRIST_Y", {0.0}},
                                                      {"L_UTHUMB", {0.0}},
                                                      {"L_LTHUMB", {0.0}},
                                                      {"L_UINDEX", {0.0}},
                                                      {"L_LINDEX", {0.0}},
                                                      {"L_ULITTLE", {0.0}},
                                                      {"L_LLITTLE", {0.0}}};
  std::vector<std::string> jvrc_filtered = {"R_UTHUMB_S",  "R_LTHUMB_S",  "R_UINDEX_S",  "R_LINDEX_S",
                                            "R_ULITTLE_S", "R_LLITTLE_S", "L_UTHUMB_S",  "L_LTHUMB_S",
                                            "L_UINDEX_S",  "L_LINDEX_S",  "L_ULITTLE_S", "L_LLITTLE_S"};
  tvm::RobotPtr jvrc = tvm::robot::fromURDF(clock, "JVRC1", jvrc_urdf, false, jvrc_filtered, ref_q);
  tvm::RobotPtr ground = tvm::robot::fromURDF(clock, "ground", ground_urdf, true, {}, {});

  auto jvrc_lf = std::make_shared<tvm::robot::Frame>(
      "LFullSoleFrame", jvrc, "L_ANKLE_P_S",
      sva::PTransformd{Eigen::Vector3d(0.014592470601201057, 0.010025011375546455, -0.138)});
  auto jvrc_rf = std::make_shared<tvm::robot::Frame>(
      "RFullSoleFrame", jvrc, "R_ANKLE_P_S",
      sva::PTransformd{Eigen::Vector3d(0.014592470601201057, 0.010025011375546455, -0.138)});
  auto ground_f = std::make_shared<tvm::robot::Frame>("GroundFrame", ground, "ground", sva::PTransformd::Identity());

  auto jvrc_lh = std::make_shared<tvm::robot::Frame>("LeftHand", jvrc, "L_WRIST_Y_S", sva::PTransformd::Identity());

  auto jvrc_rh = std::make_shared<tvm::robot::Frame>("RightHand", jvrc, "R_WRIST_Y_S", sva::PTransformd::Identity());

  auto jvrc_chest = std::make_shared<tvm::robot::Frame>("Chest", jvrc, "WAIST_R_S", sva::PTransformd::Identity());

  auto jvrc_relbow =
      std::make_shared<tvm::robot::Frame>("RightElbow", jvrc, "R_ELBOW_P_S", sva::PTransformd::Identity());

  auto jvrc_body = std::make_shared<tvm::robot::Frame>("Body", jvrc, "PELVIS_S", sva::PTransformd::Identity());

  auto contact_lf_ground = std::make_shared<tvm::robot::Contact>(
      jvrc_lf, ground_f,
      std::vector<sva::PTransformd>{{Eigen::Vector3d(0.1093074306845665, -0.06831501424312592, 0.)},
                                    {Eigen::Vector3d(0.10640743374824524, 0.06836499273777008, 0.)},
                                    {Eigen::Vector3d(-0.10778241604566574, 0.06897497922182083, 0.)},
                                    {Eigen::Vector3d(-0.1079324409365654, -0.069024957716465, 0.)}});
  auto contact_rf_ground = std::make_shared<tvm::robot::Contact>(
      jvrc_rf, ground_f,
      std::vector<sva::PTransformd>{{Eigen::Vector3d(-0.1079324409365654, 0.069024957716465, 0.)},
                                    {Eigen::Vector3d(-0.10778241604566574, -0.06897497922182083, 0.)},
                                    {Eigen::Vector3d(0.10640743374824524, -0.06836499273777008, 0.)},
                                    {Eigen::Vector3d(0.1093074306845665, 0.06831501424312592, 0.)}});

  auto dyn_fn = std::make_shared<tvm::robot::internal::DynamicFunction>(jvrc);
  auto lfg_fn =
      std::make_shared<tvm::robot::internal::GeometricContactFunction>(contact_lf_ground, Eigen::Matrix6d::Identity());
  dyn_fn->addContact(contact_lf_ground, true, 0.7, 4);
  auto rfg_fn =
      std::make_shared<tvm::robot::internal::GeometricContactFunction>(contact_rf_ground, Eigen::Matrix6d::Identity());
  dyn_fn->addContact(contact_rf_ground, true, 0.7, 4);

  auto com_in_fn = std::make_shared<tvm::robot::CoMInConvexFunction>(jvrc);
  auto cube = makeCube(jvrc->com(), 0.05);
  for(auto p : cube)
  {
    com_in_fn->addPlane(p);
  }

  auto posture_fn = std::make_shared<tvm::robot::PostureFunction>(jvrc);
  std::string joint = "NECK_P";
  std::vector<double> joint_q = {1.5};
  posture_fn->posture(joint, joint_q);
  /* This target configuration induces a collision between R_WRIST_Y_S and WAIST_R_S */
  posture_fn->posture("R_SHOULDER_P", {0.});
  posture_fn->posture("R_SHOULDER_R", {0.});
  posture_fn->posture("R_SHOULDER_Y", {1.3});
  posture_fn->posture("R_ELBOW_P", {-2.2});
  auto com_fn = std::make_shared<tvm::robot::CoMFunction>(jvrc);
  com_fn->com(com_fn->com() + Eigen::Vector3d(0, 0, -0.1));
  auto ori_fn = std::make_shared<tvm::robot::OrientationFunction>(jvrc_lh);
  ori_fn->orientation(sva::RotY(-tvm::constant::pi / 2));
  auto pos_fn = std::make_shared<tvm::robot::PositionFunction>(jvrc_lh);
  pos_fn->position(pos_fn->position() + Eigen::Vector3d{0.3, -0.1, 0.2});

  std::shared_ptr<tvm::robot::JointsSelector> ori_js =
      tvm::robot::JointsSelector::InactiveJoints(ori_fn, jvrc, {"R_ELBOW_P"});
  std::shared_ptr<tvm::robot::JointsSelector> pos_js =
      tvm::robot::JointsSelector::InactiveJoints(pos_fn, jvrc, {"R_ELBOW_P"});

  auto collision_fn = std::make_shared<tvm::robot::CollisionFunction>(clock);
  auto rhConvex = loadConvex(jvrc_rh);
  auto chestConvex = loadConvex(jvrc_chest);
  auto bodyConvex = loadConvex(jvrc_body);
  auto relbowConvex = loadConvex(jvrc_relbow);
  collision_fn->addCollision(rhConvex, chestConvex);
  collision_fn->addCollision(relbowConvex, bodyConvex);

  pb.add(lfg_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(0)});
  pb.add(rfg_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(0)});
  auto tdyn = pb.add(dyn_fn == 0., tvm::task_dynamics::None(), {tvm::requirements::PriorityLevel(0)});
  dyn_fn->addPositiveLambdaToProblem(pb);

  pb.add(posture_fn == 0., tvm::task_dynamics::PD(1.),
         {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(1.)});
  pb.add(com_fn == 0., tvm::task_dynamics::PD(2.),
         {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(100.)});
  pb.add(ori_js == 0., tvm::task_dynamics::PD(2.),
         {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(10.)});
  pb.add(pos_js == 0., tvm::task_dynamics::PD(1.),
         {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(10.)});

  pb.add(collision_fn >= 0., tvm::task_dynamics::VelocityDamper(dt, {0.1, 0.055, 0}, tvm::constant::big_number),
         {tvm::requirements::PriorityLevel(0)});
  pb.add(com_in_fn >= 0., tvm::task_dynamics::VelocityDamper(dt, {0.005, 0.001, 0}, tvm::constant::big_number),
         {tvm::requirements::PriorityLevel(0)});

  /* Bounds */
  pb.add(jvrc->lQBound() <= jvrc->qJoints() <= jvrc->uQBound(),
         tvm::task_dynamics::VelocityDamper(dt, {0.01, 0.001, 0}, tvm::constant::big_number),
         {tvm::requirements::PriorityLevel(0)});
  pb.add(jvrc->lTauBound() <= jvrc->tau() <= jvrc->uTauBound(), tvm::task_dynamics::None(),
         {tvm::requirements::PriorityLevel(0)});

  tvm::LinearizedControlProblem lpb(pb);

  lpb.add(tvm::hint::Substitution(lpb.constraint(tdyn.get()), jvrc->tau()));

  tvm::scheme::WeightedLeastSquares solver(tvm::solver::DefaultLSSolverOptions{});

  /** The position of the frame should not change */
  auto X_0_lf_init = jvrc->mbc().bodyPosW[jvrc->mb().bodyIndexByName("L_ANKLE_P_S")];
  auto X_0_rf_init = jvrc->mbc().bodyPosW[jvrc->mb().bodyIndexByName("R_ANKLE_P_S")];

  RobotPublisher rpub("/control/");
  RobotPublisher epub("/control/env_1/");
  std::cout << "Will run solver for " << iter << " iterations" << std::endl;
  size_t i = 0;
  tvm::utils::set_is_malloc_allowed(false);
  for(i = 0; i < iter; ++i)
  {
    bool b = solver.solve(lpb);
    clock.advance();
    if(!b)
    {
      break;
    }
    rpub.publish(*jvrc);
    epub.publish(*ground);
    auto X_0_lf = jvrc->mbc().bodyPosW[jvrc->mb().bodyIndexByName("L_ANKLE_P_S")];
    auto X_0_rf = jvrc->mbc().bodyPosW[jvrc->mb().bodyIndexByName("R_ANKLE_P_S")];
    auto error_lf = sva::transformError(X_0_lf, X_0_lf_init).vector().norm();
    CHECK(error_lf < 1e-3);
    auto error_rf = sva::transformError(X_0_rf, X_0_rf_init).vector().norm();
    CHECK(error_rf < 1e-3);
    CHECK(collision_fn->value()(0) >= 0.05);
    for(int i = 0; i < com_in_fn->size(); ++i)
    {
      CHECK(com_in_fn->value()(i) > 0);
    }
  }
  tvm::utils::set_is_malloc_allowed(true);
  CHECK(i == iter);
  auto lastQ = jvrc->qJoints()->value();
  for(int i = 0; i < lastQ.size(); ++i)
  {
    CHECK(jvrc->uQBound()(i) >= lastQ(i));
    CHECK(jvrc->lQBound()(i) <= lastQ(i));
  }
}
#endif
