/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Robot.h>
#include <tvm/robot/PositionFunction.h>

namespace tvm
{

namespace robot
{

PositionFunction::PositionFunction(FramePtr frame) : function::abstract::Function(3), frame_(frame)
{
  reset();
  // clang-format off
  registerUpdates(Update::Value, &PositionFunction::updateValue,
                  Update::Velocity, &PositionFunction::updateVelocity,
                  Update::Jacobian, &PositionFunction::updateJacobian,
                  Update::NormalAcceleration, &PositionFunction::updateNormalAcceleration);
  // clang-format on
  addOutputDependency<PositionFunction>(Output::Value, Update::Value);
  addOutputDependency<PositionFunction>(Output::Velocity, Update::Velocity);
  addOutputDependency<PositionFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<PositionFunction>(Output::NormalAcceleration, Update::NormalAcceleration);
  const auto & robot = frame_->robot();
  addVariable(robot.q(), false);
  addInputDependency<PositionFunction>(Update::Value, frame_, Frame::Output::Position);
  addInputDependency<PositionFunction>(Update::Velocity, frame_, Frame::Output::Velocity);
  addInputDependency<PositionFunction>(Update::Jacobian, frame_, Frame::Output::Jacobian);
  addInputDependency<PositionFunction>(Update::NormalAcceleration, frame_, Frame::Output::NormalAcceleration);
}

void PositionFunction::reset() { pos_ = frame_->position().translation(); }

void PositionFunction::updateValue() { value_ = frame_->position().translation() - pos_; }

void PositionFunction::updateVelocity() { velocity_ = frame_->velocity().linear(); }

void PositionFunction::updateJacobian() { splitJacobian(frame_->jacobian().bottomRows<3>(), frame_->robot().q()); }

void PositionFunction::updateNormalAcceleration() { normalAcceleration_ = frame_->normalAcceleration().linear(); }

} // namespace robot

} // namespace tvm
