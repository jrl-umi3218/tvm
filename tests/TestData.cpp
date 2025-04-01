/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/graph/CallGraph.h>
#include <tvm/graph/abstract/Node.h>

#include <iostream>

#include <benchmark/benchmark.h>

#include <Eigen/Core>

struct Derived : public tvm::graph::abstract::Outputs
{
  SET_OUTPUTS(Derived, O0, O1, O2)
};

struct Derived2 : public Derived
{
  SET_OUTPUTS(Derived2, O3)
};

struct Derived3 : public Derived2
{};

struct Derived4 : public Derived3
{
  SET_OUTPUTS(Derived4, O4, O5, O6, O7, O8)
};

struct Derived5 : public Derived4
{
  DISABLE_OUTPUTS(Output::O4, Derived::Output::O0)
};

struct Derived6 : public Derived5
{
  CLEAR_DISABLED_OUTPUTS()
};

struct AnotherOutput : public tvm::graph::abstract::Outputs
{
  SET_OUTPUTS(AnotherOutput, O0, O1)
};

struct TestInputs : public tvm::graph::internal::Inputs
{
  TestInputs(std::shared_ptr<Derived> s) { addInput(s, Derived::Output::O0, Derived::Output::O1); }
};

struct Robot : public tvm::graph::abstract::Node<Robot>
{
  SET_OUTPUTS(Robot, K1, K2, K3, V1, V2)
  SET_UPDATES(Robot, Kinematics, Velocity)

  Robot()
  {
    // clang-format off
    registerUpdates(Update::Kinematics, &Robot::updateKinematics,
                    Update::Velocity, &Robot::updateVelocity);
    // clang-format on

    addOutputDependency(Output::K1, Update::Kinematics);
    addOutputDependency(Output::K2, Update::Kinematics);
    addOutputDependency(Output::K3, Update::Kinematics);
    addOutputDependency(Output::V1, Update::Velocity);
    addOutputDependency(Output::V2, Update::Velocity);

    addInternalDependency(Update::Velocity, Update::Kinematics);
  }

  virtual void updateKinematics() { k = k.transpose(); }

  virtual void updateVelocity() { v = v * v; }

  Eigen::Matrix3d k = Eigen::Matrix3d::Random();
  Eigen::Matrix3d v = Eigen::Matrix3d::Random();
};

struct Robot2 : public Robot
{
  SET_OUTPUTS(Robot2, D1, D2)
  SET_UPDATES(Robot2, Dynamics)

  Robot2()
  {
    registerUpdates(Update::Dynamics, &Robot2::updateDynamics);

    addOutputDependency<Robot2>(Output::D1, Update::Dynamics);
    addOutputDependency<Robot2>(Output::D2, Update::Dynamics);

    addInternalDependency<Robot2>(Update::Dynamics, Robot::Update::Velocity);
  }

  void updateKinematics() override { k = k * k; }

  virtual void updateDynamics() { d = d * d; }

  Eigen::Matrix3d d = Eigen::Matrix3d::Random();
};

struct RobotFunction : public tvm::graph::abstract::Node<RobotFunction>
{};

void compile_check()
{
  static_assert(Derived::OutputSize == 3, "");
  static_assert(Derived2::OutputSize == 4, "");
  static_assert(Derived3::OutputSize == 4, "");
  static_assert(Derived4::OutputSize == 9, "");
  static_assert(tvm::graph::abstract::is_valid_output<Derived3>(Derived::Output::O2), "");
  static_assert(!tvm::graph::abstract::is_valid_output<Derived3>(AnotherOutput::Output::O1), "");
  static_assert(
      tvm::graph::abstract::is_valid_output<Derived4>(Derived4::Output::O8, Derived2::Output::O3, Derived::Output::O0),
      "");
  static_assert(!Derived5::OutputStaticallyEnabled(Derived4::Output::O4), "");
  static_assert(Derived5::OutputStaticallyEnabled(Derived4::Output::O7), "");
  static_assert(!Derived5::OutputStaticallyEnabled(Derived::Output::O0), "");
  static_assert(Derived5::OutputStaticallyEnabled(Derived::Output::O1), "");

  static_assert(Derived6::OutputStaticallyEnabled(Derived4::Output::O4), "");
  static_assert(Derived6::OutputStaticallyEnabled(Derived4::Output::O7), "");
  static_assert(Derived6::OutputStaticallyEnabled(Derived::Output::O0), "");
  static_assert(Derived6::OutputStaticallyEnabled(Derived::Output::O1), "");

  static_assert(Robot::OutputSize == 5, "");
  static_assert(Robot::UpdateSize == 2, "");
  static_assert(Robot2::OutputSize == 7, "");
  static_assert(Robot2::UpdateSize == 3, "");
}

static void BM_ManualGraph(benchmark::State & state)
{
  auto r2 = std::make_shared<Robot2>();

  auto update = [&]() {
    r2->update(static_cast<int>(Robot::Update::Kinematics));
    r2->update(static_cast<int>(Robot::Update::Velocity));
    r2->update(static_cast<int>(Robot2::Update::Dynamics));
  };
  while(state.KeepRunning())
  {
    update();
  }
}
BENCHMARK(BM_ManualGraph);

static void BM_CallGraph(benchmark::State & state)
{
  // Use the CallGraph to generate the same code
  tvm::graph::CallGraph g;
  {
    // Make sure these objects lifetime is extended beyond the scope
    auto userID = std::make_shared<tvm::graph::internal::Inputs>();
    auto r2 = std::make_shared<Robot2>();
    userID->addInput(r2, Robot2::Output::D1, Robot2::Output::D2);

    g.add(userID);
  }
  g.update();

  while(state.KeepRunning())
  {
    g.execute();
  }
}
BENCHMARK(BM_CallGraph);

BENCHMARK_MAIN();
