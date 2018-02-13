#include <tvm/robot/CollisionFunction.h>

#include <tvm/Robot.h>
#include <tvm/utils/sch.h>

namespace tvm
{

namespace robot
{

CollisionFunction::CollisionFunction(double dt,
  FramePtr f1, std::shared_ptr<sch::S_Object> o1, const sva::PTransformd & X_f1_o1,
  FramePtr f2, std::shared_ptr<sch::S_Object> o2, const sva::PTransformd & X_f2_o2)
: function::abstract::Function(1),
  dt_(dt),
  o1_(o1), o2_(o2),
  pair_(new sch::CD_Pair(o1.get(), o2.get()))
{
  const auto & r1 = f1->robot();
  const auto & r2 = f2->robot();
  int r1Dof = r1.mb().nrDof();
  int r2Dof = r2.mb().nrDof();
  assert(r1Dof >= r2Dof);

  registerUpdates(Update::Value, &CollisionFunction::updateValue,
                  Update::Velocity, &CollisionFunction::updateVelocity,
                  Update::Jacobian, &CollisionFunction::updateJacobian,
                  Update::NormalAcceleration, &CollisionFunction::updateNormalAcceleration);
  addOutputDependency<CollisionFunction>(Output::Value, Update::Value);
  addOutputDependency<CollisionFunction>(Output::Velocity, Update::Velocity);
  addOutputDependency<CollisionFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<CollisionFunction>(Output::NormalAcceleration, Update::NormalAcceleration);

  addInternalDependency<CollisionFunction>(Update::Jacobian, Update::Value);
  addInternalDependency<CollisionFunction>(Update::Velocity, Update::Jacobian);
  addInternalDependency<CollisionFunction>(Update::NormalAcceleration, Update::Jacobian);

  if(r1Dof > 0)
  {
    addInputDependency<CollisionFunction>(Update::Value, f1, Frame::Output::Position);
    addInputDependency<CollisionFunction>(Update::Jacobian, f1->robotPtr(), Robot::Output::Dynamics);
    addInputDependency<CollisionFunction>(Update::NormalAcceleration, f1->robotPtr(), Robot::Output::Acceleration);
    objects_.push_back({f1, o1.get(), X_f1_o1, Eigen::Vector3d::Zero(), f1->rbdJacobian()});
    addVariable(r1.q(), false);
  }
  if(r2Dof > 0)
  {
    addInputDependency<CollisionFunction>(Update::Value, f2, Frame::Output::Position);
    addInputDependency<CollisionFunction>(Update::Jacobian, f2->robotPtr(), Robot::Output::Dynamics);
    addInputDependency<CollisionFunction>(Update::NormalAcceleration, f2->robotPtr(), Robot::Output::Acceleration);
    objects_.push_back({f2, o2.get(), X_f2_o2, Eigen::Vector3d::Zero(), f2->rbdJacobian()});
    if(r1.qFreeFlyer() != r2.qFreeFlyer())
    {
      addVariable(r2.q(), false);
    }
  }

  fullJac_.resize(1, r1Dof);
  distJac_.resize(1, r1Dof);
}

void CollisionFunction::updateValue()
{
  for(const auto & o : objects_)
  {
    tvm::utils::transform(*o.o_, o.X_f_o_ * o.f_->position());
  }

  double dist = tvm::utils::distance(*pair_, closestPoints_[0], closestPoints_[1]);
  dist = dist >= 0 ? std::sqrt(dist) : -std::sqrt(-dist);

  for(size_t i = 0; i < objects_.size(); ++i)
  {
    auto & o = objects_[i];
    o.nearestPoint_ = (sva::PTransformd(closestPoints_[i])*o.f_->position().inv()).translation();
    o.jac_.point(o.nearestPoint_);
  }

  value_(0) = dist;
}

void CollisionFunction::updateJacobian()
{
  normVecDist_ = (closestPoints_[0] - closestPoints_[1]) / value_(0);
  double sign = 1.;
  for(auto & o : objects_)
  {
    const auto & r = o.f_->robot();
    jacobian_[r.qFreeFlyer().get()].setZero();
    jacobian_[r.qJoints().get()].setZero();
  }
  for(auto & o : objects_)
  {
    const auto & r = o.f_->robot();
    const Eigen::MatrixXd & jac = o.jac_.jacobian(r.mb(), r.mbc());
    distJac_.block(0, 0, 1, o.jac_.dof()).noalias() =
      (sign *  normVecDist_).transpose() * jac.block(3, 0, 3, o.jac_.dof());
    o.jac_.fullJacobian(r.mb(),
                        distJac_.block(0, 0, 1, o.jac_.dof()),
                        fullJac_);
    //FIXME Should check for qFreeFlyer and/or qJoints nullity?
    jacobian_[r.qFreeFlyer().get()] += fullJac_.block(0, 0, 1, r.qFreeFlyer()->space().tSize());
    jacobian_[r.qJoints().get()] += fullJac_.block(0, r.qFreeFlyer()->space().tSize(), 1, r.qJoints()->space().tSize());
    sign *= -1.;
  }
}

void CollisionFunction::updateVelocity()
{
  velocity_(0) = 0.;
  double sign = 1.;
  for(auto & o : objects_)
  {
    const auto & r = o.f_->robot();
    Eigen::Vector3d pSpeed = o.jac_.velocity(r.mb(), r.mbc()).linear();
    velocity_(0) += sign * pSpeed.dot(normVecDist_);
    sign *= -1.;
  }
}

void CollisionFunction::updateNormalAcceleration()
{
  Eigen::Vector3d speedVec = (normVecDist_ - prevNormVecDist_) / dt_;
  prevNormVecDist_ = normVecDist_;
  normalAcceleration_(0) = 0.;
  double sign = 1.;
  for(auto & o : objects_)
  {
    const auto & r = o.f_->robot();
    Eigen::Vector3d pNormalAcc = o.jac_.normalAcceleration(r.mb(), r.mbc(), r.normalAccB()).linear();
    Eigen::Vector3d pSpeed = o.jac_.velocity(r.mb(), r.mbc()).linear();
    normalAcceleration_(0) += sign * ( pNormalAcc.dot(normVecDist_) + pSpeed.dot(speedVec) );
    sign *= -1.;
  }
}

} // namespace robot

} // namespace tvm
