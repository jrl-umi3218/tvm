#include <tvm/robot/Frame.h>

#include <tvm/Robot.h>

namespace
{
  Eigen::Matrix3d hat(const Eigen::Vector3d & v)
  {
    Eigen::Matrix3d ret;
    ret << 0., -v.z(), v.y(),
           v.z(), 0., -v.x(),
           -v.y(), v.x(), 0.;
    return ret;
  }
}

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
  jacTmp_(6, jac_.dof()),
  jacobian_(6, robot->mb().nrDof()) // FIXME Don't allocate until needed?
{
  registerUpdates(
                  Update::Position, &Frame::updatePosition,
                  Update::Jacobian, &Frame::updateJacobian,
                  Update::Velocity, &Frame::updateVelocity,
                  Update::NormalAcceleration, &Frame::updateNormalAcceleration
                 );


  addOutputDependency(Output::Position, Update::Position);
  addInputDependency(Update::Position, robot_, Robot::Output::Kinematics);

  addOutputDependency(Output::Jacobian, Update::Jacobian);
  addInternalDependency(Update::Jacobian, Update::Position);
  addInputDependency(Update::Jacobian, robot_, Robot::Output::Dynamics);

  addOutputDependency(Output::Velocity, Update::Velocity);
  addInputDependency(Update::Velocity, robot_, Robot::Output::Dynamics);

  addOutputDependency(Output::NormalAcceleration, Update::NormalAcceleration);
  addInputDependency(Update::NormalAcceleration, robot_, Robot::Output::Acceleration);

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
                                          robot_->mbc());
  jacTmp_ = partialJac;
  Eigen::Matrix3d h = -hat(robot_->mbc().bodyPosW[bodyId_].rotation().transpose() * X_b_f_.translation());
  jacTmp_.block(3, 0, 3, jac_.dof()).noalias() += h * partialJac.block(3, 0, 3, jac_.dof());
  jac_.fullJacobian(robot_->mb(), jacTmp_, jacobian_);
}

void Frame::updateVelocity()
{
  velocity_ = X_b_f_ * robot_->mbc().bodyVelW[bodyId_];
}

void Frame::updateNormalAcceleration()
{
  normalAcceleration_ = X_b_f_ * jac_.normalAcceleration(robot_->mb(), robot_->mbc(), robot_->normalAccB());
}

const std::string & Frame::body() const
{
  return robot_->mb().body(bodyId_).name();
}

}

}
