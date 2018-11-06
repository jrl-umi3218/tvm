#include <tvm/robot/OrientationFunction.h>
#include <tvm/Robot.h>

namespace tvm
{

namespace robot
{


OrientationFunction::OrientationFunction(FramePtr frame)
: function::abstract::Function(3),
  frame_(frame)
{
  reset();
  registerUpdates(Update::Value, &OrientationFunction::updateValue,
                  Update::Velocity, &OrientationFunction::updateVelocity,
                  Update::Jacobian, &OrientationFunction::updateJacobian,
                  Update::NormalAcceleration, &OrientationFunction::updateNormalAcceleration);
  addOutputDependency<OrientationFunction>(Output::Value, Update::Value);
  addOutputDependency<OrientationFunction>(Output::Velocity, Update::Velocity);
  addOutputDependency<OrientationFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<OrientationFunction>(Output::NormalAcceleration, Update::NormalAcceleration);
  const auto & robot = frame_->robot();
  addVariable(robot.q(), false);
  addInputDependency<OrientationFunction>(Update::Value, frame_, Frame::Output::Position);
  addInputDependency<OrientationFunction>(Update::Velocity, frame_, Frame::Output::Velocity);
  addInputDependency<OrientationFunction>(Update::Jacobian, frame_, Frame::Output::Jacobian);
  addInputDependency<OrientationFunction>(Update::NormalAcceleration, frame_, Frame::Output::NormalAcceleration);
}

void OrientationFunction::reset()
{
  ori_ = frame_->position().rotation();
}

void OrientationFunction::updateValue()
{
  value_ = sva::rotationError(ori_, frame_->position().rotation());
}

void OrientationFunction::updateVelocity()
{
  velocity_ = frame_->velocity().angular();
}

void OrientationFunction::updateJacobian()
{
  splitJacobian(frame_->jacobian().topRows<3>(), frame_->robot().q());
}

void OrientationFunction::updateNormalAcceleration()
{
  normalAcceleration_ = frame_->normalAcceleration().angular();
}

} // namespace robot

} // namespace tvm
