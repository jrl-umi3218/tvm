#include <tvm/robot/CoMFunction.h>

namespace tvm
{

namespace robot
{

CoMFunction::CoMFunction(RobotPtr robot)
: function::abstract::Function(3),
  robot_(robot),
  jac_(robot_->mb())
{
  reset();
  registerUpdates(Update::Value, &CoMFunction::updateValue,
                  Update::Velocity, &CoMFunction::updateVelocity,
                  Update::Jacobian, &CoMFunction::updateJacobian,
                  Update::NormalAcceleration, &CoMFunction::updateNormalAcceleration,
                  Update::JDot, &CoMFunction::updateJDot);
  addOutputDependency<CoMFunction>(Output::Value, Update::Value);
  addOutputDependency<CoMFunction>(Output::Velocity, Update::Velocity);
  addOutputDependency<CoMFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<CoMFunction>(Output::NormalAcceleration, Update::NormalAcceleration);
  addOutputDependency<CoMFunction>(Output::JDot, Update::JDot);
  addVariable(robot_->q(), false);
  addInputDependency<CoMFunction>(Update::Value, robot_, Robot::Output::CoM);
  addInputDependency<CoMFunction>(Update::Velocity, robot_, Robot::Output::Dynamics);
  addInputDependency<CoMFunction>(Update::Jacobian, robot_, Robot::Output::Dynamics);
  addInputDependency<CoMFunction>(Update::NormalAcceleration, robot_, Robot::Output::Acceleration);
  addInputDependency<CoMFunction>(Update::JDot, robot_, Robot::Output::Acceleration);
}

void CoMFunction::reset()
{
  com_ = robot_->com();
}

void CoMFunction::updateValue()
{
  value_ = robot_->com() - com_;
}

void CoMFunction::updateVelocity()
{
  velocity_ = jac_.velocity(robot_->mb(), robot_->mbc());
}

void CoMFunction::updateJacobian()
{
  splitJacobian(jac_.jacobian(robot_->mb(), robot_->mbc()), robot_->q());
}

void CoMFunction::updateNormalAcceleration()
{
  normalAcceleration_ = jac_.normalAcceleration(robot_->mb(), robot_->mbc(), robot_->normalAccB());
}

void CoMFunction::updateJDot()
{
  splitJacobian(jac_.jacobianDot(robot_->mb(), robot_->mbc()), robot_->q());
}

} // namespace robot

} // namespace tvm
