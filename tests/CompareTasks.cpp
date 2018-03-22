/* Copyright 2017-2018 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <benchmark/benchmark.h>

#include <mc_rbdyn_urdf/urdf.h>

#include <RBDyn/ID.h>

#include <fstream>
#include <iostream>

static std::string hrp2_path = HRP2_DRC_DESCRIPTION_PATH;
static std::string env_path = MC_ENV_DESCRIPTION_PATH;
static std::string hrp2_urdf = hrp2_path + "/urdf/hrp2drc.urdf";
static std::string ground_urdf = env_path + "/urdf/ground.urdf";
static std::string hrp2_convex_path = hrp2_path + "/convex/hrp2_drc/";

static double dt = 0.005;

/* Tasks */

#include <Tasks/Bounds.h>
#include <Tasks/QPSolver.h>
#include <Tasks/QPConstr.h>
#include <Tasks/QPContacts.h>
#include <Tasks/QPContactConstr.h>
#include <Tasks/QPMotionConstr.h>
#include <Tasks/QPTasks.h>

#include <RBDyn/CoM.h>
#include <RBDyn/EulerIntegration.h>
#include <RBDyn/FK.h>
#include <RBDyn/FV.h>
#include <RBDyn/FA.h>

static void BM_Tasks(benchmark::State & state)
{
  std::vector<rbd::MultiBody> mbs;
  std::vector<rbd::MultiBodyConfig> mbcs;

  auto load_robot = [&mbs,&mbcs](const std::string & path, bool fixed,
                       const std::vector<std::string> & filteredLinks,
                       const std::vector<std::vector<double>> & ref_q)
    -> mc_rbdyn_urdf::Limits
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
    rbd::forwardKinematics(data.mb, data.mbc);
    rbd::forwardVelocity(data.mb, data.mbc);
    mbs.push_back(data.mb);
    mbcs.push_back(data.mbc);
    return data.limits;
  };
  mc_rbdyn_urdf::Limits hrp2_limits;
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
  std::vector<std::vector<double>> ref_q = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.773}, {0.0}, {0.0}, {-0.4537856055185257}, {0.8726646259971648}, {-0.41887902047863906}, {0.0}, {0.0}, {0.0}, {-0.4537856055185257}, {0.8726646259971648}, {-0.41887902047863906}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.7853981633974483}, {-0.3490658503988659}, {0.0}, {-1.3089969389957472}, {0.0}, {0.0}, {0.0}, {0.3490658503988659}, {0.7853981633974483}, {0.3490658503988659}, {0.0}, {-1.0}, {0.0}, {0.0}, {0.0}, {0.3490658503988659}};
  hrp2_limits = load_robot(hrp2_urdf, false, hrp2_filtered, ref_q);
  load_robot(ground_urdf, true, {}, {});

  tasks::qp::QPSolver solver;

  tasks::qp::PositiveLambda posLambda{};
  posLambda.addToSolver(mbs, solver);

  auto bound2TasksBound = [](const rbd::MultiBody & mb,
                             const std::map<std::string, std::vector<double>> & b,
                             double mul,
                             double ff_limit)
  {
    std::vector<std::vector<double>> ret;
    for(const auto & j : mb.joints())
    {
      if(b.count(j.name()))
      {
        ret.push_back(b.at(j.name()));
        for(auto & ji : ret.back())
        {
          ji *= mul;
        }
      }
      else
      {
        if(j.dof() != 0 && j.dof() != 6)
        {
          std::cerr << "Missing limit for " << j.name() << std::endl;
          throw(std::runtime_error("Missing limit"));
        }
        if(j.dof() == 0)
        {
          ret.push_back({});
        }
        else
        {
          ret.push_back(std::vector<double>(6, ff_limit));
        }
      }
    }
    return ret;
  };
  tasks::TorqueBound tb {
    bound2TasksBound(mbs[0], hrp2_limits.torque, -1, 0),
    bound2TasksBound(mbs[0], hrp2_limits.torque, 1, 0)
  };
  tasks::qp::MotionConstr motionConstr{mbs, 0, tb};
  motionConstr.addToSolver(mbs, solver);

  tasks::QBound qb {
    bound2TasksBound(mbs[0], hrp2_limits.lower, 1, -std::numeric_limits<double>::infinity()),
    bound2TasksBound(mbs[0], hrp2_limits.upper, 1, std::numeric_limits<double>::infinity())
  };
  tasks::AlphaBound ab {
    bound2TasksBound(mbs[0], hrp2_limits.velocity, -0.5, -std::numeric_limits<double>::infinity()),
    bound2TasksBound(mbs[0], hrp2_limits.velocity, 0.5, std::numeric_limits<double>::infinity())
  };
  tasks::qp::DamperJointLimitsConstr jl { mbs, 0, qb, ab, 0.1, 0.01, 0.5, dt};
  jl.addToSolver(mbs, solver);

  tasks::qp::ContactPosConstr contactConstr{dt};
  contactConstr.addToSolver(mbs, solver);

  auto ll5_pos = mbcs[0].bodyPosW[mbs[0].bodyIndexByName("LLEG_LINK5")];
  Eigen::Vector3d ll5_o {0.014592470601201057, 0.010025011375546455, -0.138};
  auto rl5_pos = mbcs[0].bodyPosW[mbs[0].bodyIndexByName("RLEG_LINK5")];
  Eigen::Vector3d rl5_o {0.014592470601201057, 0.010025011375546455, -0.138};
  auto g_pos = mbcs[1].bodyPosW[mbs[1].bodyIndexByName("ground")];
  std::vector<tasks::qp::UnilateralContact> uniC {
    {0, 1, "LLEG_LINK5", "ground",
      {
        ll5_o + Eigen::Vector3d(0.1093074306845665, -0.06831501424312592, 0.),
        ll5_o + Eigen::Vector3d(0.10640743374824524, 0.06836499273777008, 0.),
        ll5_o + Eigen::Vector3d(-0.10778241604566574, 0.06897497922182083, 0.),
        ll5_o + Eigen::Vector3d(-0.1079324409365654, -0.069024957716465, 0.)
      },
      Eigen::Matrix3d::Identity(),
      g_pos * ll5_pos.inv(),
      4, 0.7, {ll5_o}
    },
    {0, 1, "RLEG_LINK5", "ground",
      {
        rl5_o + Eigen::Vector3d(-0.1079324409365654, 0.069024957716465, 0.),
        rl5_o + Eigen::Vector3d(-0.10778241604566574, -0.06897497922182083, 0.),
        rl5_o + Eigen::Vector3d(0.10640743374824524, -0.06836499273777008, 0.),
        rl5_o + Eigen::Vector3d(0.1093074306845665, 0.06831501424312592, 0.)
      },
      Eigen::Matrix3d::Identity(),
      g_pos * rl5_pos.inv(),
      4, 0.7, {rl5_o}
    }
  };

  auto q_target = mbcs[0].q;
  q_target[mbs[0].jointIndexByName("HEAD_JOINT1")] = {0.75};
  tasks::qp::PostureTask pt{mbs, 0, q_target, 1., 1.};
  solver.addTask(mbs, &pt);

  auto com_target = rbd::computeCoM(mbs[0], mbcs[0]);
  com_target.z() -= 0.1;
  tasks::qp::CoMTask com_t {mbs, 0, com_target};
  tasks::qp::SetPointTask com_sp {mbs, 0, &com_t, 2., 100.};
  solver.addTask(mbs, &com_sp);

  auto lh_pos = mbcs[0].bodyPosW[mbs[0].bodyIndexByName("LARM_LINK6")];

  tasks::qp::OrientationTask ori_t {mbs, 0, "LARM_LINK6", Eigen::Quaterniond(sva::RotY(-M_PI/2))};
  tasks::qp::SetPointTask ori_sp {mbs, 0, &ori_t, 2., 10.};
  solver.addTask(mbs, &ori_sp);

  tasks::qp::PositionTask pos_t {mbs, 0, "LARM_LINK6", lh_pos.translation() + Eigen::Vector3d{0.3, -0.1, 0.2}};
  tasks::qp::SetPointTask pos_sp {mbs, 0, &pos_t, 1., 10.};
  solver.addTask(mbs, &pos_sp);

  /** Make sure everything is up-to-date in QPSolver */
  solver.nrVars(mbs, uniC, {});
  solver.updateConstrSize();
  while(state.KeepRunning())
  {
    solver.solve(mbs, mbcs);
    for(size_t i = 0; i < mbcs.size(); ++i)
    {
      const auto & mb = mbs[i];
      auto & mbc = mbcs[i];
      rbd::eulerIntegration(mb, mbc, dt);
      rbd::forwardKinematics(mb, mbc);
      rbd::forwardVelocity(mb, mbc);
      rbd::forwardAcceleration(mb, mbc);
    }
  }
}
BENCHMARK(BM_Tasks)->Unit(benchmark::kMicrosecond);//->MinTime(10.0);

/* TVM */

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/VelocityDamper.h>

#include <tvm/Robot.h>
#include <tvm/robot/internal/GeometricContactFunction.h>
#include <tvm/robot/internal/DynamicFunction.h>
#include <tvm/robot/CoMFunction.h>
#include <tvm/robot/OrientationFunction.h>
#include <tvm/robot/PositionFunction.h>
#include <tvm/robot/PostureFunction.h>
#include <tvm/robot/utils.h>

static void BM_TVM(benchmark::State & state)
{
  tvm::ControlProblem pb(dt);

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
  std::map<std::string, std::vector<double>> ref_q = {
    {"Root", {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.773}},
    {"RLEG_JOINT0", {0.0}},
    {"RLEG_JOINT1", {0.0}},
    {"RLEG_JOINT2", {-0.4537856055185257}},
    {"RLEG_JOINT3", {0.8726646259971648}},
    {"RLEG_JOINT4", {-0.41887902047863906}},
    {"RLEG_JOINT5", {0.0}},
    {"LLEG_JOINT0", {0.0}},
    {"LLEG_JOINT1", {0.0}},
    {"LLEG_JOINT2", {-0.4537856055185257}},
    {"LLEG_JOINT3", {0.8726646259971648}},
    {"LLEG_JOINT4", {-0.41887902047863906}},
    {"LLEG_JOINT5", {0.0}},
    {"CHEST_JOINT0", {0.0}},
    {"CHEST_JOINT1", {0.0}},
    {"HEAD_JOINT0", {0.0}},
    {"HEAD_JOINT1", {0.0}},
    {"RARM_JOINT0", {0.7853981633974483}},
    {"RARM_JOINT1", {-0.3490658503988659}},
    {"RARM_JOINT2", {0.0}},
    {"RARM_JOINT3", {-1.3089969389957472}},
    {"RARM_JOINT4", {0.0}},
    {"RARM_JOINT5", {0.0}},
    {"RARM_JOINT6", {0.0}},
    {"RARM_JOINT7", {0.3490658503988659}},
    {"LARM_JOINT0", {0.7853981633974483}},
    {"LARM_JOINT1", {0.3490658503988659}},
    {"LARM_JOINT2", {0.0}},
    {"LARM_JOINT3", {-1.0}},
    {"LARM_JOINT4", {0.0}},
    {"LARM_JOINT5", {0.0}},
    {"LARM_JOINT6", {0.0}},
    {"LARM_JOINT7", {0.3490658503988659}}
  };
  tvm::RobotPtr hrp2 = tvm::robot::fromURDF(pb.clock(), "HRP2", hrp2_urdf, false, hrp2_filtered, ref_q);
  tvm::RobotPtr ground = tvm::robot::fromURDF(pb.clock(), "ground", ground_urdf, true, {}, {});

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

  auto posture_fn = std::make_shared<tvm::robot::PostureFunction>(hrp2);
  std::string joint = "HEAD_JOINT1"; std::vector<double> joint_q = {1.5};
  posture_fn->posture(joint, joint_q);

  auto com_fn = std::make_shared<tvm::robot::CoMFunction>(hrp2);
  com_fn->com(com_fn->com() + Eigen::Vector3d(0, 0, -0.1));

  auto ori_fn = std::make_shared<tvm::robot::OrientationFunction>(hrp2_lh);
  ori_fn->orientation(sva::RotY(-M_PI/2));

  auto pos_fn = std::make_shared<tvm::robot::PositionFunction>(hrp2_lh);
  pos_fn->position(pos_fn->position() + Eigen::Vector3d{0.3, -0.1, 0.2});

  pb.add(lfg_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(0)});
  pb.add(rfg_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(0)});
  pb.add(dyn_fn == 0., tvm::task_dynamics::None(), {tvm::requirements::PriorityLevel(0)});
  dyn_fn->addPositiveLambdaToProblem(pb);

  pb.add(posture_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(1.)});
  pb.add(com_fn == 0., tvm::task_dynamics::PD(2.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(100.)});
  pb.add(ori_fn == 0., tvm::task_dynamics::PD(2.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(10.)});
  pb.add(pos_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(10.)});

  /* Bounds */
  pb.add(hrp2->lQBound() <= hrp2->qJoints() <= hrp2->uQBound(), tvm::task_dynamics::VelocityDamper(dt, {0.01, 0.001, 0}, tvm::constant::big_number), { tvm::requirements::PriorityLevel(0) });
  pb.add(hrp2->lTauBound() <= hrp2->tau() <= hrp2->uTauBound(), tvm::task_dynamics::None(), { tvm::requirements::PriorityLevel(0) });

  tvm::LinearizedControlProblem lpb(pb);

  tvm::scheme::WeightedLeastSquares solver(false);

  while(state.KeepRunning())
  {
    solver.solve(lpb);
  }
}
BENCHMARK(BM_TVM)->Unit(benchmark::kMicrosecond);//->MinTime(10.0);

BENCHMARK_MAIN()
