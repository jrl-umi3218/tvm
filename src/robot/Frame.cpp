#include <tvm/robot/Frame.h>

#include <tvm/Robot.h>

namespace tvm
{

namespace robot
{

Frame::Frame(std::string name,
             RobotPtr robot,
             const std::string & body,
             sva::PTransformd X_b_f)
: name_(std::move(name)),
  robot_(robot),
  bodyId_(robot->mb().bodyIndexByName(body)),
  jac_(robot->mb(), body),
  X_b_f_(std::move(X_b_f)),
  jacobian_(6, robot->mb().nrDof()) // FIXME Don't allocate until needed?
{
  registerUpdates(
                  Update::Position, &Frame::updatePosition,
                  Update::Jacobian, &Frame::updateJacobian,
                  Update::Velocity, &Frame::updateVelocity,
                  Update::NormalAcceleration, &Frame::updateNormalAcceleration
                 );

  /** Position depends on the robot's state */
  addOutputDependency(Output::Position, Update::Position);
  addInputDependency(Update::Position, robot_, Robot::Output::q);
  /** Jacobian needs the frame position */
  addOutputDependency(Output::Jacobian, Update::Jacobian);
  addInternalDependency(Update::Jacobian, Update::Position);
  /** Velocity only depends on the state */
  addOutputDependency(Output::Velocity, Update::Velocity);
  addInputDependency(Update::Velocity, robot_, Robot::Output::q);
  /** NormalAcceleration only depends on the state */
  addOutputDependency(Output::NormalAcceleration, Update::NormalAcceleration);
  addInputDependency(Update::NormalAcceleration, robot_, Robot::Output::q);

  /** Initialize all data */
  updatePosition();
  updateJacobian();
  updateVelocity();
  updateNormalAcceleration();
}

void Frame::updatePosition()
{
  const auto & X_0_b = robot_->mbc().bodyPosW[bodyId_];
  /** X_0_f */
  position_ = X_b_f_ * X_0_b;
}

void Frame::updateJacobian()
{
  assert(jacobian_.rows() == 6 && jacobian_.cols() == robot_->mb().nrDof());
  const auto & partialJac = jac_.jacobian(robot_->mb(),
                                          robot_->mbc(),
                                          position_);
  jac_.fullJacobian(robot_->mb(), partialJac, jacobian_);
}

void Frame::updateVelocity()
{
  velocity_ = jac_.velocity(robot_->mb(), robot_->mbc(), X_b_f_);
}

void Frame::updateNormalAcceleration()
{
  normalAcceleration_ = jac_.normalAcceleration(robot_->mb(), robot_->mbc(), robot_->normalAccB(), X_b_f_, sva::MotionVecd(Eigen::Vector6d::Zero()));
}

}

}
