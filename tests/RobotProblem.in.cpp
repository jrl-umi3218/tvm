#include <tvm/Robot.h>
#include <tvm/robot/internal/ContactFunction.h>
#include <tvm/Task.h>

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/scheme/WeightedLeastSquares.h>

#include <mc_rbdyn_urdf/urdf.h>

#include <fstream>
#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

static std::string hrp2_urdf = "@HRP2_DRC_DESCRIPTION_PATH@/urdf/hrp2drc.urdf";
static std::string ground_urdf = "@MC_ENV_DESCRIPTION_PATH@/urdf/ground.urdf";

TEST_CASE("Test a problem with a robot")
{
  auto load_robot = [](const std::string & name, const std::string & path, bool fixed)
  {
    std::ifstream ifs(path);
    if(!ifs.good())
    {
      std::cerr << "Failed to open " << path << std::endl;
      std::exit(1);
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    auto data = mc_rbdyn_urdf::rbdyn_from_urdf(ss.str(), fixed);
    return std::make_shared<tvm::Robot>(name, data.mb, data.mbc);
  };
  auto hrp2 = load_robot("HRP2", hrp2_urdf, false);
  {
    auto q = hrp2->mbc().q;
    std::vector<std::vector<double>> ref_q = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.773}, {0.0}, {0.0}, {-0.4537856055185257}, {0.8726646259971648}, {-0.41887902047863906}, {0.0}, {0.0}, {0.0}, {-0.4537856055185257}, {0.8726646259971648}, {-0.41887902047863906}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.7853981633974483}, {-0.3490658503988659}, {0.0}, {-1.3089969389957472}, {0.0}, {0.0}, {0.0}, {0.3490658503988659}, {-0.3490658503988659}, {0.3490658503988659}, {-0.3490658503988659}, {0.3490658503988659}, {-0.3490658503988659}, {0.7853981633974483}, {0.3490658503988659}, {0.0}, {-1.3089969389957472}, {0.0}, {0.0}, {0.0}, {0.3490658503988659}, {-0.3490658503988659}, {0.3490658503988659}, {-0.3490658503988659}, {0.3490658503988659}, {-0.3490658503988659}};
    size_t j = 0;
    for(size_t i = 0; i < q.size(); ++i)
    {
      if(q[i].size())
      {
        if(ref_q[j].size() != q[i].size())
        {
          std::cerr << "SOMETHING IS WRONG" << std::endl;
          std::exit(1);
        }
        q[i] = ref_q[j];
        ++j;
      }
    }
  }
  auto ground = load_robot("ground", ground_urdf, false);

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
  auto ground_f = std::make_shared<tvm::robot::Frame>("frame",
                                                      ground,
                                                      "ground",
                                                      sva::PTransformd::Identity());

  auto contact_lf_ground = std::make_shared<tvm::robot::Contact>
    (hrp2_lf, ground_f, std::vector<sva::PTransformd>{sva::PTransformd::Identity()});
  auto contact_rf_ground = std::make_shared<tvm::robot::Contact>
    (hrp2_rf, ground_f, std::vector<sva::PTransformd>{sva::PTransformd::Identity()});

  auto lfg_fn = std::make_shared<tvm::robot::internal::ContactFunction>(contact_lf_ground, Eigen::Matrix6d::Identity());
  auto rfg_fn = std::make_shared<tvm::robot::internal::ContactFunction>(contact_rf_ground, Eigen::Matrix6d::Identity());

  tvm::ControlProblem pb;
  pb.add(lfg_fn == 0., std::make_shared<tvm::task_dynamics::ProportionalDerivative>(0), {tvm::requirements::PriorityLevel(0)});
  pb.add(rfg_fn == 0., std::make_shared<tvm::task_dynamics::ProportionalDerivative>(0), {tvm::requirements::PriorityLevel(0)});

  tvm::LinearizedControlProblem lpb(pb);

  tvm::scheme::WeightedLeastSquares solver(false);
  /** The position of the frame should not change */
  auto X_0_lf_init = hrp2->mbc().bodyPosW[hrp2->mb().bodyIndexByName("LLEG_LINK5")];
  auto X_0_rf_init = hrp2->mbc().bodyPosW[hrp2->mb().bodyIndexByName("RLEG_LINK5")];
  for(size_t i = 0; i < 1000; ++i)
  {
    solver.solve(lpb);
    hrp2->updateTimeDependency(0.005);
    auto X_0_lf = hrp2->mbc().bodyPosW[hrp2->mb().bodyIndexByName("LLEG_LINK5")];
    auto X_0_rf = hrp2->mbc().bodyPosW[hrp2->mb().bodyIndexByName("RLEG_LINK5")];
    auto error_lf = sva::transformError(X_0_lf, X_0_lf_init).vector().norm();
    FAST_CHECK_UNARY(error_lf < 1e-12);
    auto error_rf = sva::transformError(X_0_rf, X_0_rf_init).vector().norm();
    FAST_CHECK_UNARY(error_rf < 1e-12);
  }
}
