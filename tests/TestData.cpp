#include <tvm/data/Node.h>

#include <iostream>

struct Derived : public tvm::data::Outputs
{
  SET_OUTPUTS(Derived, O0, O1, O2)
};

struct Derived2 : public Derived
{
  SET_OUTPUTS(Derived2, O3)
};

struct Derived3 : public Derived2
{
};

struct Derived4 : public Derived3
{
  SET_OUTPUTS(Derived4, O4, O5, O6, O7, O8)
};

struct AnotherOutput : public tvm::data::Outputs
{
  SET_OUTPUTS(AnotherOutput, O0, O1)
};

struct TestInputs : public tvm::data::Inputs
{
  TestInputs(std::shared_ptr<Derived> s)
  {
    addInput(s, Derived::Output::O0, Derived::Output::O1);
  }
};

struct Robot : public tvm::data::Node<Robot>
{
  SET_OUTPUTS(Robot, K1, K2, K3, V1, V2)
  SET_UPDATES(Robot, Kinematics, Velocity)

  Robot()
  {
    registerUpdates(Update::Kinematics, &Robot::updateKinematics,
                    Update::Velocity, &Robot::updateVelocity);

    addOutputDependency(Output::K1, Update::Kinematics);
    addOutputDependency(Output::K2, Update::Kinematics);
    addOutputDependency(Output::K3, Update::Kinematics);
    addOutputDependency(Output::V1, Update::Velocity);
    addOutputDependency(Output::V2, Update::Velocity);

    addInternalDependency(Update::Velocity, Update::Kinematics);
  }

  virtual void updateKinematics()
  {
    std::cout << "Robot::updateKinematics()" << std::endl;
  }

  virtual void updateVelocity()
  {
    std::cout << "Robot::updateVelocity()" << std::endl;
  }
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

  void updateKinematics() override
  {
    std::cout << "Robot2::updateKinematics()" << std::endl;
  }

  virtual void updateDynamics()
  {
    std::cout << "Robot2::updateDynamics()" << std::endl;
  }
};

struct RobotFunction : public tvm::data::Node<RobotFunction>
{
};

int main()
{
  static_assert(Derived::OutputSize == 3, "");
  static_assert(Derived2::OutputSize == 4, "");
  static_assert(Derived3::OutputSize == 4, "");
  static_assert(Derived4::OutputSize == 9, "");
  static_assert(tvm::data::is_valid_output<Derived3>(Derived::Output::O2), "");
  static_assert(!tvm::data::is_valid_output<Derived3>(AnotherOutput::Output::O1), "");
  static_assert(tvm::data::is_valid_output<Derived4>(Derived4::Output::O8, Derived2::Output::O3, Derived::Output::O0), "");

  static_assert(Robot::OutputSize == 5, "");
  static_assert(Robot::UpdateSize == 2, "");
  static_assert(Robot2::OutputSize == 7, "");
  static_assert(Robot2::UpdateSize == 3, "");

  Robot r;
  Robot2 r2;

  return 0;
}
