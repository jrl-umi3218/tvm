#include <tvm/robot/CoMInConvexFunction.h>

namespace tvm
{

namespace robot
{

CoMInConvexFunction::CoMInConvexFunction(RobotPtr robot)
: robot_(robot),
  jac_(robot_->mb())
{
  registerUpdates(
                  Update::Value, &CoMInConvexFunction::updateValue,
                  Update::Velocity, &CoMInConvexFunction::updateVelocity,
                  Update::Jacobian, &CoMInConvexFunction::updateJacobian,
                  Update::NormalAcceleration, &CoMInConvexFunction::updateNormalAcceleration
                 );
  addOutputDependency<CoMInConvexFunction>(Output::Value, Update::Value);
  addOutputDependency<CoMInConvexFunction>(Output::Velocity, Update::Velocity);
  addOutputDependency<CoMInConvexFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<CoMInConvexFunction>(Output::NormalAcceleration, Update::NormalAcceleration);
  addVariable(robot_->q(), false);
  addInputDependency<CoMInConvexFunction>(Update::Value, *robot_, Robot::Output::CoM);
  addInputDependency<CoMInConvexFunction>(Update::Velocity, *robot_, Robot::Output::Dynamics);
  addInputDependency<CoMInConvexFunction>(Update::Jacobian, *robot_, Robot::Output::Dynamics);
  addInputDependency<CoMInConvexFunction>(Update::NormalAcceleration, *robot_, Robot::Output::Acceleration);
  addInternalDependency<CoMInConvexFunction>(Update::Velocity, Update::Jacobian);
  addInternalDependency<CoMInConvexFunction>(Update::NormalAcceleration, Update::Jacobian);
}

void CoMInConvexFunction::addPlane(geometry::PlanePtr plane)
{
  planes_.push_back(plane);
  addInputDependency<CoMInConvexFunction>(Update::Value, *plane, geometry::Plane::Output::Position);
  addInputDependency<CoMInConvexFunction>(Update::Velocity, *plane, geometry::Plane::Output::Velocity);
  addInputDependency<CoMInConvexFunction>(Update::NormalAcceleration, *plane, geometry::Plane::Output::Acceleration);
  resize(planes_.size());
}

void CoMInConvexFunction::reset()
{
  planes_.resize(0);
  resize(0);
}

void CoMInConvexFunction::updateValue()
{
  Eigen::DenseIndex i = 0;
  for(const auto & p : planes_)
  {
    value_(i++) = robot_->com().dot(p->normal()) + p->offset();
  }
}

void CoMInConvexFunction::updateVelocity()
{
  Eigen::DenseIndex i = 0;
  comSpeed_ = jac_.velocity(robot_->mb(), robot_->mbc());
  for(const auto & p : planes_)
  {
    velocity_(i++) = p->normal().dot(comSpeed_ - p->speed()) +
                     p->normalDot().dot(robot_->com() - p->point());
  }
}

void CoMInConvexFunction::updateJacobian()
{
  Eigen::DenseIndex i = 0;
  const Eigen::MatrixXd & jac = jac_.jacobian(robot_->mb(), robot_->mbc());
  const auto & qFF = robot_->qFreeFlyer(); int qFFSize = qFF->space().tSize();
  const auto & qJoints = robot_->qJoints(); int qJointsSize = qJoints->space().tSize();
  for(const auto & p : planes_)
  {
    if(qFFSize)
    {
      jacobian_[qFF.get()].block(i, 0, 1, qFFSize).noalias() = p->normal().transpose() * jac.middleCols(0, qFFSize);
    }
    if(qJointsSize)
    {
      jacobian_[qJoints.get()].block(i, 0, 1, qJointsSize).noalias() = p->normal().transpose() * jac.middleCols(qFFSize, qJointsSize);
    }
    ++i;
  }
}

void CoMInConvexFunction::updateNormalAcceleration()
{
  Eigen::DenseIndex i = 0;
  Eigen::Vector3d comNAcc = jac_.normalAcceleration(robot_->mb(), robot_->mbc());
  for(const auto & p : planes_)
  {
    normalAcceleration_(i++) =
      p->normal().dot(comNAcc - p->acceleration()) +
      2 * p->normalDot().dot(comSpeed_ - p->speed()) +
      p->normalDotDot().dot(robot_->com() - p->point());
  }
}

} // namespace robot

} // namespace tvm
