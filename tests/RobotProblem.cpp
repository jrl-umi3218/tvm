#include <tvm/Robot.h>
#include <tvm/robot/internal/GeometricContactFunction.h>
#include <tvm/robot/internal/DynamicFunction.h>
#include <tvm/robot/CollisionFunction.h>
#include <tvm/robot/CoMFunction.h>
#include <tvm/robot/CoMInConvexFunction.h>
#include <tvm/robot/ConvexHull.h>
#include <tvm/robot/JointsSelector.h>
#include <tvm/robot/OrientationFunction.h>
#include <tvm/robot/PositionFunction.h>
#include <tvm/robot/PostureFunction.h>
#include <tvm/Task.h>

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/VelocityDamper.h>
#include <tvm/utils/sch.h>

#include <mc_rbdyn_urdf/urdf.h>

#include <RBDyn/ID.h>

#include <fstream>
#include <iostream>

#include "RobotPublisher.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

static std::string hrp2_path = HRP2_DRC_DESCRIPTION_PATH;
static std::string env_path = MC_ENV_DESCRIPTION_PATH;
static std::string hrp2_urdf = hrp2_path + "/urdf/hrp2drc.urdf";
static std::string ground_urdf = env_path + "/urdf/ground.urdf";
static std::string hrp2_convex_path = hrp2_path + "/convex/hrp2_drc/";

tvm::robot::ConvexHullPtr loadConvex(tvm::robot::FramePtr f)
{
  return std::make_shared<tvm::robot::ConvexHull>
  (
    hrp2_convex_path + f->body() + "-ch.txt",
    f, sva::PTransformd::Identity()
  );
}

/** Build a cube as a set of planes from a given origin and size */
std::vector<tvm::geometry::PlanePtr> makeCube(const Eigen::Vector3d & origin, double size)
{
  return {
    std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{1, 0, 0}, origin + Eigen::Vector3d{-size, 0, 0}),
    std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{-1, 0, 0}, origin + Eigen::Vector3d{size, 0, 0}),
    std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{0, 1, 0}, origin + Eigen::Vector3d{0, -size, 0}),
    std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{0, -1, 0}, origin + Eigen::Vector3d{0, size, 0}),
    std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{0, 0, 1}, origin + Eigen::Vector3d{0, 0, -size}),
    std::make_shared<tvm::geometry::Plane>(Eigen::Vector3d{0, 0, -1}, origin + Eigen::Vector3d{0, 0, size})
  };
}

TEST_CASE("Test a problem with a robot")
{
  size_t iter = 2000;
  double dt = 0.005;

  tvm::ControlProblem pb(dt);

  auto load_robot = [](tvm::Clock & clock,
                       const std::string & name, const std::string & path, bool fixed,
                       const std::vector<std::string> & filteredLinks,
                       const std::vector<std::vector<double>> & ref_q)
    -> std::pair<tvm::RobotPtr, mc_rbdyn_urdf::Limits>
  {
    std::ifstream ifs(path);
    if(!ifs.good())
    {
      std::cerr << "Failed to open " << path << std::endl;
      std::exit(1);
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    auto data = mc_rbdyn_urdf::rbdyn_from_urdf(ss.str(), fixed, filteredLinks);
    data.mbc.gravity = Eigen::Vector3d(0, 0, 9.81);
    auto init_q = data.mbc.q;
    size_t j = 0;
    for(size_t i = 0; i < init_q.size(); ++i)
    {
      if(init_q[i].size())
      {
        if(ref_q[j].size() != init_q[i].size())
        {
          std::cerr << "SOMETHING IS WRONG" << std::endl;
          std::exit(1);
        }
        init_q[i] = ref_q[j];
        ++j;
      }
    }
    if(init_q.size())
    {
      data.mbc.q = init_q;
    }
    return {std::make_shared<tvm::Robot>(clock, name, data.mbg, data.mb, data.mbc), data.limits};
  };
  tvm::RobotPtr hrp2; mc_rbdyn_urdf::Limits hrp2_limits;
  std::vector<std::string> hrp2_filtered = {};
  for(size_t i = 0; i < 5; ++i)
  {
    {
    std::stringstream ss;
    ss << "LHAND_LINK" << i;
    hrp2_filtered.push_back(ss.str());
    }
    {
    std::stringstream ss;
    ss << "RHAND_LINK" << i;
    hrp2_filtered.push_back(ss.str());
    }
  }
  std::vector<std::vector<double>> ref_q = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.773}, {0.0}, {0.0}, {-0.4537856055185257}, {0.8726646259971648}, {-0.41887902047863906}, {0.0}, {0.0}, {0.0}, {-0.4537856055185257}, {0.8726646259971648}, {-0.41887902047863906}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.7853981633974483}, {-0.3490658503988659}, {0.0}, {-1.3089969389957472}, {0.0}, {0.0}, {0.0}, {0.3490658503988659}, {-0.3490658503988659}, {0.3490658503988659}, {-0.3490658503988659}, {0.3490658503988659}, {-0.3490658503988659}, {0.7853981633974483}, {0.3490658503988659}, {0.0}, {-1.0}, {0.0}, {0.0}, {0.0}, {0.3490658503988659}, {-0.3490658503988659}, {0.3490658503988659}, {-0.3490658503988659}, {0.3490658503988659}, {-0.3490658503988659}};
  std::tie(hrp2, hrp2_limits) = load_robot(pb.clock(), "HRP2", hrp2_urdf, false, {}, ref_q);
  tvm::RobotPtr ground;
  std::tie(ground, std::ignore) = load_robot(pb.clock(), "ground", ground_urdf, true, {}, {});

  auto hrp2_lf = std::make_shared<tvm::robot::Frame>("LFullSoleFrame",
                                                     hrp2,
                                                     "LLEG_LINK5",
                                                     sva::PTransformd{Eigen::Vector3d(0.014592470601201057, 0.010025011375546455, -0.138)}
                                                     );
  auto hrp2_rf = std::make_shared<tvm::robot::Frame>("RFullSoleFrame",
                                                     hrp2,
                                                     "RLEG_LINK5",
                                                     sva::PTransformd{Eigen::Vector3d(0.014592470601201057, 0.010025011375546455, -0.138)}
                                                     );
  auto ground_f = std::make_shared<tvm::robot::Frame>("GroundFrame",
                                                      ground,
                                                      "ground",
                                                      sva::PTransformd::Identity());

  auto hrp2_lh = std::make_shared<tvm::robot::Frame>("LeftHand",
                                                     hrp2,
                                                     "LARM_LINK6",
                                                     sva::PTransformd::Identity());

  auto hrp2_rh = std::make_shared<tvm::robot::Frame>("RightHand",
                                                     hrp2,
                                                     "RARM_LINK6",
                                                     sva::PTransformd::Identity());

  auto hrp2_chest = std::make_shared<tvm::robot::Frame>("Chest",
                                                        hrp2,
                                                        "CHEST_LINK1",
                                                        sva::PTransformd::Identity());

  auto hrp2_relbow = std::make_shared<tvm::robot::Frame>("RightElbow",
                                                         hrp2,
                                                         "RARM_LINK3",
                                                         sva::PTransformd::Identity());

  auto hrp2_body = std::make_shared<tvm::robot::Frame>("Body",
                                                       hrp2,
                                                       "BODY",
                                                       sva::PTransformd::Identity());

  auto contact_lf_ground = std::make_shared<tvm::robot::Contact>
    (hrp2_lf, ground_f, std::vector<sva::PTransformd>{
        {Eigen::Vector3d(0.1093074306845665, -0.06831501424312592, 0.)},
        {Eigen::Vector3d(0.10640743374824524, 0.06836499273777008, 0.)},
        {Eigen::Vector3d(-0.10778241604566574, 0.06897497922182083, 0.)},
        {Eigen::Vector3d(-0.1079324409365654, -0.069024957716465, 0.)}
     });
  auto contact_rf_ground = std::make_shared<tvm::robot::Contact>
    (hrp2_rf, ground_f, std::vector<sva::PTransformd>{
        {Eigen::Vector3d(-0.1079324409365654, 0.069024957716465, 0.)},
        {Eigen::Vector3d(-0.10778241604566574, -0.06897497922182083, 0.)},
        {Eigen::Vector3d(0.10640743374824524, -0.06836499273777008, 0.)},
        {Eigen::Vector3d(0.1093074306845665, 0.06831501424312592, 0.)}
     });

  auto dyn_fn = std::make_shared<tvm::robot::internal::DynamicFunction>(hrp2);
  auto lfg_fn = std::make_shared<tvm::robot::internal::GeometricContactFunction>(contact_lf_ground, Eigen::Matrix6d::Identity());
  dyn_fn->addContact(contact_lf_ground, true, 0.7, 4);
  auto rfg_fn = std::make_shared<tvm::robot::internal::GeometricContactFunction>(contact_rf_ground, Eigen::Matrix6d::Identity());
  dyn_fn->addContact(contact_rf_ground, true, 0.7, 4);

  auto com_in_fn = std::make_shared<tvm::robot::CoMInConvexFunction>(hrp2);
  auto cube = makeCube(hrp2->com(), 0.05);
  for(auto p : cube) { com_in_fn->addPlane(p); }

  auto posture_fn = std::make_shared<tvm::robot::PostureFunction>(hrp2);
  std::string joint = "HEAD_JOINT1"; std::vector<double> joint_q = {1.5};
  posture_fn->posture(joint, joint_q);
  /* This target configuration induces a collision between RARM_LINK6 and CHEST_LINK1 */
  posture_fn->posture("RARM_JOINT0", {0.});
  posture_fn->posture("RARM_JOINT1", {0.});
  posture_fn->posture("RARM_JOINT2", {1.3});
  posture_fn->posture("RARM_JOINT3", {-2.2});
  auto com_fn = std::make_shared<tvm::robot::CoMFunction>(hrp2);
  com_fn->com(com_fn->com() + Eigen::Vector3d(0, 0, -0.1));
  auto ori_fn = std::make_shared<tvm::robot::OrientationFunction>(hrp2_lh);
  ori_fn->orientation(sva::RotY(-M_PI/2));
  auto pos_fn = std::make_shared<tvm::robot::PositionFunction>(hrp2_lh);
  pos_fn->position(pos_fn->position() + Eigen::Vector3d{0.3, -0.1, 0.2});

  std::shared_ptr<tvm::robot::JointsSelector> ori_js = tvm::robot::JointsSelector::UnactiveJoints(ori_fn, hrp2, {"LARM_JOINT3"});
  std::shared_ptr<tvm::robot::JointsSelector> pos_js = tvm::robot::JointsSelector::UnactiveJoints(pos_fn, hrp2, {"LARM_JOINT3"});

  auto collision_fn = std::make_shared<tvm::robot::CollisionFunction>(pb.clock());
  auto rhConvex = loadConvex(hrp2_rh);
  auto chestConvex = loadConvex(hrp2_chest);
  auto bodyConvex = loadConvex(hrp2_body);
  auto relbowConvex = loadConvex(hrp2_relbow);
  collision_fn->addCollision(rhConvex, chestConvex);
  collision_fn->addCollision(relbowConvex, bodyConvex);

  pb.add(lfg_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(0)});
  pb.add(rfg_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(0)});
  pb.add(dyn_fn == 0., tvm::task_dynamics::None(), {tvm::requirements::PriorityLevel(0)});
  dyn_fn->addPositiveLambdaToProblem(pb);

  pb.add(posture_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(1.)});
  pb.add(com_fn == 0., tvm::task_dynamics::PD(2.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(100.)});
  pb.add(ori_js == 0., tvm::task_dynamics::PD(2.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(10.)});
  pb.add(pos_js == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(10.)});

  pb.add(collision_fn >= 0., tvm::task_dynamics::VelocityDamper(dt, {0.1, 0.055, 0}, tvm::constant::big_number), {tvm::requirements::PriorityLevel(0)});
  pb.add(com_in_fn >= 0., tvm::task_dynamics::VelocityDamper(dt, {0.005, 0.001, 0}, tvm::constant::big_number), {tvm::requirements::PriorityLevel(0)});

  /* Bounds */
  Eigen::VectorXd lq(hrp2->mb().nrParams());
  lq.setConstant(-tvm::constant::big_number);
  Eigen::VectorXd uq(hrp2->mb().nrParams());
  uq.setConstant(tvm::constant::big_number);
  Eigen::VectorXd lqd(hrp2->mb().nrDof());
  Eigen::VectorXd uqd(hrp2->mb().nrDof());
  Eigen::VectorXd ltau(hrp2->mb().nrDof());
  Eigen::VectorXd utau(hrp2->mb().nrDof());
  ltau.head(6).setConstant(0);
  utau.head(6).setConstant(0);
  for(const auto & j : hrp2_limits.lower)
  {
    const auto & mb = hrp2->mb();
    if(j.second.size() == 0) { continue; }
    lq(mb.jointPosInParam(mb.jointIndexByName(j.first))) = j.second[0];
    uq(mb.jointPosInParam(mb.jointIndexByName(j.first))) = hrp2_limits.upper.at(j.first)[0];
    lqd(mb.jointPosInDof(mb.jointIndexByName(j.first))) = -hrp2_limits.velocity.at(j.first)[0];
    uqd(mb.jointPosInDof(mb.jointIndexByName(j.first))) = hrp2_limits.velocity.at(j.first)[0];
    ltau(mb.jointPosInDof(mb.jointIndexByName(j.first))) = -hrp2_limits.torque.at(j.first)[0];
    utau(mb.jointPosInDof(mb.jointIndexByName(j.first))) = hrp2_limits.torque.at(j.first)[0];
  }
  Eigen::VectorXd tmp = lq.tail(hrp2->mb().nrParams() - 7);
  lq = tmp;
  tmp = uq.tail(hrp2->mb().nrParams() - 7);
  uq = tmp;
  CHECK(lq.size() == uq.size());
  for(int i = 0; i < lq.size(); ++i)
  {
    CHECK(lq(i) < uq(i));
  }
  pb.add(lq <= hrp2->qJoints() <= uq, tvm::task_dynamics::VelocityDamper(dt, {0.01, 0.001, 0}, tvm::constant::big_number), { tvm::requirements::PriorityLevel(0) });
  pb.add(ltau <= hrp2->tau() <= utau, tvm::task_dynamics::None(), { tvm::requirements::PriorityLevel(0) });

  tvm::LinearizedControlProblem lpb(pb);

  tvm::scheme::WeightedLeastSquares solver(false);

  /** The position of the frame should not change */
  auto X_0_lf_init = hrp2->mbc().bodyPosW[hrp2->mb().bodyIndexByName("LLEG_LINK5")];
  auto X_0_rf_init = hrp2->mbc().bodyPosW[hrp2->mb().bodyIndexByName("RLEG_LINK5")];

  RobotPublisher rpub("/control/");
  RobotPublisher epub("/control/env_1/");
  std::cout << "Will run solver for " << iter << " iterations" << std::endl;
  size_t i = 0;
  for(i = 0; i < iter; ++i)
  {
    bool b = solver.solve(lpb);
    if(!b) { break; }
    rpub.publish(*hrp2);
    epub.publish(*ground);
    auto X_0_lf = hrp2->mbc().bodyPosW[hrp2->mb().bodyIndexByName("LLEG_LINK5")];
    auto X_0_rf = hrp2->mbc().bodyPosW[hrp2->mb().bodyIndexByName("RLEG_LINK5")];
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
  CHECK(i == iter);
  auto lastQ = hrp2->qJoints()->value();
  CHECK(uq.size() == lastQ.size());
  CHECK(lq.size() == lastQ.size());
  for(int i = 0; i < uq.size(); ++i)
  {
    CHECK(uq(i) >= lastQ(i));
    CHECK(lq(i) <= lastQ(i));
  }
}
