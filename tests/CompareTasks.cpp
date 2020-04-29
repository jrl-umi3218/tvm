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

#include <RBDyn/parsers/urdf.h>

#include <RBDyn/ID.h>

#include <fstream>
#include <iostream>

static std::string jvrc_path = JVRC_DESCRIPTION_PATH;
static std::string env_path = MC_ENV_DESCRIPTION_PATH;
static std::string jvrc_urdf = jvrc_path + "/urdf/jvrc1.urdf";
static std::string ground_urdf = env_path + "/urdf/ground.urdf";
static std::string jvrc_convex_path = jvrc_path + "/convex/jvrc1/";

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

#ifndef M_PI
#  define M_PI 3.141592653589793238462643383279502884e+00 // from boost/math/constant
#endif

static void BM_Tasks(benchmark::State & state)
{
  std::vector<rbd::MultiBody> mbs;
  std::vector<rbd::MultiBodyConfig> mbcs;

  auto load_robot = [&mbs,&mbcs](const std::string & path, bool fixed,
                       const std::vector<std::string> & filteredLinks,
                       const std::vector<std::vector<double>> & ref_q)
    -> rbd::parsers::Limits
  {
    std::ifstream ifs(path);
    if(!ifs.good())
    {
      std::cerr << "Failed to open " << path << std::endl;
      std::exit(1);
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    auto data = rbd::parsers::from_urdf(ss.str(), fixed, filteredLinks);
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
  rbd::parsers::Limits jvrc_limits;
  std::vector<std::string> jvrc_filtered = {
    "R_UTHUMB_S", "R_LTHUMB_S", "R_UINDEX_S", "R_LINDEX_S", "R_ULITTLE_S", "R_LLITTLE_S",
    "L_UTHUMB_S", "L_LTHUMB_S", "L_UINDEX_S", "L_LINDEX_S", "L_ULITTLE_S", "L_LLITTLE_S"};
  std::vector<std::vector<double>> ref_q = {{1.0,0.0,0.0,0.0,0.0,0.0,0.8275}, {-0.38}, {-0.01}, {0.0}, {0.72}, {-0.01}, {-0.33}, {-0.38}, {0.02}, {0.0}, {0.72}, {-0.02}, {-0.33}, {0.0}, {0.13}, {0.0}, {0.0}, {0.0}, {0.0}, {-0.052}, {-0.17}, {0.0}, {-0.52}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {-0.052}, {0.17}, {0.0}, {-0.52}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}};
  jvrc_limits = load_robot(jvrc_urdf, false, jvrc_filtered, ref_q);
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
    bound2TasksBound(mbs[0], jvrc_limits.torque, -1, 0),
    bound2TasksBound(mbs[0], jvrc_limits.torque, 1, 0)
  };
  tasks::qp::MotionConstr motionConstr{mbs, 0, tb};
  motionConstr.addToSolver(mbs, solver);

  tasks::QBound qb {
    bound2TasksBound(mbs[0], jvrc_limits.lower, 1, -std::numeric_limits<double>::infinity()),
    bound2TasksBound(mbs[0], jvrc_limits.upper, 1, std::numeric_limits<double>::infinity())
  };
  tasks::AlphaBound ab {
    bound2TasksBound(mbs[0], jvrc_limits.velocity, -0.5, -std::numeric_limits<double>::infinity()),
    bound2TasksBound(mbs[0], jvrc_limits.velocity, 0.5, std::numeric_limits<double>::infinity())
  };
  tasks::qp::DamperJointLimitsConstr jl { mbs, 0, qb, ab, 0.1, 0.01, 0.5, dt};
  jl.addToSolver(mbs, solver);

  tasks::qp::ContactPosConstr contactConstr{dt};
  contactConstr.addToSolver(mbs, solver);

  auto ll5_pos = mbcs[0].bodyPosW[mbs[0].bodyIndexByName("L_ANKLE_P_S")];
  Eigen::Vector3d ll5_o {0.014592470601201057, 0.010025011375546455, -0.138};
  auto rl5_pos = mbcs[0].bodyPosW[mbs[0].bodyIndexByName("R_ANKLE_P_S")];
  Eigen::Vector3d rl5_o {0.014592470601201057, 0.010025011375546455, -0.138};
  auto g_pos = mbcs[1].bodyPosW[mbs[1].bodyIndexByName("ground")];
  std::vector<tasks::qp::UnilateralContact> uniC {
    {0, 1, "L_ANKLE_P_S", "ground",
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
    {0, 1, "R_ANKLE_P_S", "ground",
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
  q_target[mbs[0].jointIndexByName("NECK_P")] = {0.75};
  tasks::qp::PostureTask pt{mbs, 0, q_target, 1., 1.};
  solver.addTask(mbs, &pt);

  auto com_target = rbd::computeCoM(mbs[0], mbcs[0]);
  com_target.z() -= 0.1;
  tasks::qp::CoMTask com_t {mbs, 0, com_target};
  tasks::qp::SetPointTask com_sp {mbs, 0, &com_t, 2., 100.};
  solver.addTask(mbs, &com_sp);

  auto lh_pos = mbcs[0].bodyPosW[mbs[0].bodyIndexByName("L_WRIST_Y_S")];

  tasks::qp::OrientationTask ori_t {mbs, 0, "L_WRIST_Y_S", Eigen::Quaterniond(sva::RotY(-M_PI/2))};
  tasks::qp::SetPointTask ori_sp {mbs, 0, &ori_t, 2., 10.};
  solver.addTask(mbs, &ori_sp);

  tasks::qp::PositionTask pos_t {mbs, 0, "L_WRIST_Y_S", lh_pos.translation() + Eigen::Vector3d{0.3, -0.1, 0.2}};
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
#include <tvm/solver/defaultLeastSquareSolver.h>
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
  tvm::ControlProblem pb;
  tvm::Clock clock(dt);

  std::vector<std::string> jvrc_filtered = {
    "R_UTHUMB_S", "R_LTHUMB_S", "R_UINDEX_S", "R_LINDEX_S", "R_ULITTLE_S", "R_LLITTLE_S",
    "L_UTHUMB_S", "L_LTHUMB_S", "L_UINDEX_S", "L_LINDEX_S", "L_ULITTLE_S", "L_LLITTLE_S"};
  std::map<std::string, std::vector<double>> ref_q = {
    {"Root", {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8275}},
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
    {"L_LLITTLE", {0.0}}
  };
  tvm::RobotPtr jvrc = tvm::robot::fromURDF(clock, "JVRC1", jvrc_urdf, false, jvrc_filtered, ref_q);
  tvm::RobotPtr ground = tvm::robot::fromURDF(clock, "ground", ground_urdf, true, {}, {});

  auto jvrc_lf = std::make_shared<tvm::robot::Frame>("LFullSoleFrame",
                                                     jvrc,
                                                     "L_ANKLE_P_S",
                                                     sva::PTransformd{Eigen::Vector3d(0.014592470601201057, 0.010025011375546455, -0.138)}
                                                     );
  auto jvrc_rf = std::make_shared<tvm::robot::Frame>("RFullSoleFrame",
                                                     jvrc,
                                                     "R_ANKLE_P_S",
                                                     sva::PTransformd{Eigen::Vector3d(0.014592470601201057, 0.010025011375546455, -0.138)}
                                                     );
  auto ground_f = std::make_shared<tvm::robot::Frame>("GroundFrame",
                                                      ground,
                                                      "ground",
                                                      sva::PTransformd::Identity());

  auto jvrc_lh = std::make_shared<tvm::robot::Frame>("LeftHand",
                                                     jvrc,
                                                     "L_WRIST_Y_S",
                                                     sva::PTransformd::Identity());

  auto contact_lf_ground = std::make_shared<tvm::robot::Contact>
    (jvrc_lf, ground_f, std::vector<sva::PTransformd>{
        {Eigen::Vector3d(0.1093074306845665, -0.06831501424312592, 0.)},
        {Eigen::Vector3d(0.10640743374824524, 0.06836499273777008, 0.)},
        {Eigen::Vector3d(-0.10778241604566574, 0.06897497922182083, 0.)},
        {Eigen::Vector3d(-0.1079324409365654, -0.069024957716465, 0.)}
     });
  auto contact_rf_ground = std::make_shared<tvm::robot::Contact>
    (jvrc_rf, ground_f, std::vector<sva::PTransformd>{
        {Eigen::Vector3d(-0.1079324409365654, 0.069024957716465, 0.)},
        {Eigen::Vector3d(-0.10778241604566574, -0.06897497922182083, 0.)},
        {Eigen::Vector3d(0.10640743374824524, -0.06836499273777008, 0.)},
        {Eigen::Vector3d(0.1093074306845665, 0.06831501424312592, 0.)}
     });

  auto dyn_fn = std::make_shared<tvm::robot::internal::DynamicFunction>(jvrc);

  auto lfg_fn = std::make_shared<tvm::robot::internal::GeometricContactFunction>(contact_lf_ground, Eigen::Matrix6d::Identity());
  dyn_fn->addContact(contact_lf_ground, true, 0.7, 4);

  auto rfg_fn = std::make_shared<tvm::robot::internal::GeometricContactFunction>(contact_rf_ground, Eigen::Matrix6d::Identity());
  dyn_fn->addContact(contact_rf_ground, true, 0.7, 4);

  auto posture_fn = std::make_shared<tvm::robot::PostureFunction>(jvrc);
  std::string joint = "NECK_P"; std::vector<double> joint_q = {1.5};
  posture_fn->posture(joint, joint_q);

  auto com_fn = std::make_shared<tvm::robot::CoMFunction>(jvrc);
  com_fn->com(com_fn->com() + Eigen::Vector3d(0, 0, -0.1));

  auto ori_fn = std::make_shared<tvm::robot::OrientationFunction>(jvrc_lh);
  ori_fn->orientation(sva::RotY(-M_PI/2));

  auto pos_fn = std::make_shared<tvm::robot::PositionFunction>(jvrc_lh);
  pos_fn->position(pos_fn->position() + Eigen::Vector3d{0.3, -0.1, 0.2});

  pb.add(lfg_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(0)});
  pb.add(rfg_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(0)});
  auto dyn_task = pb.add(dyn_fn == 0., tvm::task_dynamics::None(), {tvm::requirements::PriorityLevel(0)});
  dyn_fn->addPositiveLambdaToProblem(pb);

  pb.add(posture_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(1.)});
  pb.add(com_fn == 0., tvm::task_dynamics::PD(2.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(100.)});
  pb.add(ori_fn == 0., tvm::task_dynamics::PD(2.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(10.)});
  pb.add(pos_fn == 0., tvm::task_dynamics::PD(1.), {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(10.)});

  /* Bounds */
  pb.add(jvrc->lQBound() <= jvrc->qJoints() <= jvrc->uQBound(), tvm::task_dynamics::VelocityDamper(dt, {0.01, 0.001, 0}, tvm::constant::big_number), { tvm::requirements::PriorityLevel(0) });
  pb.add(jvrc->lTauBound() <= jvrc->tau() <= jvrc->uTauBound(), tvm::task_dynamics::None(), { tvm::requirements::PriorityLevel(0) });

  tvm::LinearizedControlProblem lpb(pb);

  lpb.add(tvm::hint::Substitution(lpb.constraint(dyn_task.get()), jvrc->tau()));

  tvm::scheme::WeightedLeastSquares solver(tvm::solver::DefaultLSSolverFactory{});

  while(state.KeepRunning())
  {
    solver.solve(lpb);
    clock.advance();
  }
}
BENCHMARK(BM_TVM)->Unit(benchmark::kMicrosecond);//->MinTime(10.0);

BENCHMARK_MAIN()
