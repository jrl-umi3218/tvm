#include <tvm/robot/CollisionFunction.h>

#include <tvm/Robot.h>
#include <tvm/utils/sch.h>

namespace tvm
{

namespace robot
{

CollisionFunction::CollisionFunction(double dt)
: function::abstract::Function(0),
  dt_(dt)
{
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
}

void CollisionFunction::addCollision(
  FramePtr f1, std::shared_ptr<sch::S_Object> o1, const sva::PTransformd & X_f1_o1,
  FramePtr f2, std::shared_ptr<sch::S_Object> o2, const sva::PTransformd & X_f2_o2)
{
  const auto & r1 = f1->robot();
  const auto & r2 = f2->robot();
  int r1Dof = r1.mb().nrDof();
  int r2Dof = r2.mb().nrDof();
  if(r2Dof > r1Dof)
  {
    addCollision(f2, o2, X_f2_o2, f1, o1, X_f1_o1);
    return;
  }
  assert(r1Dof >= r2Dof);
  colls_.emplace_back(*this, f1, o1, X_f1_o1, f2, o2, X_f2_o2);
  resize(colls_.size());
  int maxDof = std::max(fullJac_.cols(), static_cast<Eigen::DenseIndex>(r1Dof));
  fullJac_.resize(1, maxDof);
  distJac_.resize(1, maxDof);
}

CollisionFunction::CollisionData::CollisionData(CollisionFunction & fn,
  FramePtr f1, std::shared_ptr<sch::S_Object> o1, const sva::PTransformd & X_f1_o1,
  FramePtr f2, std::shared_ptr<sch::S_Object> o2, const sva::PTransformd & X_f2_o2)
: o1_(o1), o2_(o2), pair_(new sch::CD_Pair(o1.get(), o2.get()))
{
  const auto & r1 = f1->robot();
  const auto & r2 = f2->robot();
  int r1Dof = r1.mb().nrDof();
  int r2Dof = r2.mb().nrDof();
  if(r1Dof > 0)
  {
    fn.addInputDependency<CollisionFunction>(Update::Value, f1, Frame::Output::Position);
    fn.addInputDependency<CollisionFunction>(Update::Jacobian, f1->robotPtr(), Robot::Output::Dynamics);
    fn.addInputDependency<CollisionFunction>(Update::NormalAcceleration, f1->robotPtr(), Robot::Output::Acceleration);
    objects_.push_back({f1, o1.get(), X_f1_o1, Eigen::Vector3d::Zero(), f1->rbdJacobian()});
    fn.addVariable(r1.q(), false);
  }
  if(r2Dof > 0)
  {
    fn.addInputDependency<CollisionFunction>(Update::Value, f2, Frame::Output::Position);
    fn.addInputDependency<CollisionFunction>(Update::Jacobian, f2->robotPtr(), Robot::Output::Dynamics);
    fn.addInputDependency<CollisionFunction>(Update::NormalAcceleration, f2->robotPtr(), Robot::Output::Acceleration);
    objects_.push_back({f2, o2.get(), X_f2_o2, Eigen::Vector3d::Zero(), f2->rbdJacobian()});
    fn.addVariable(r2.q(), false);
  }

}

void CollisionFunction::updateValue()
{
  Eigen::DenseIndex i = 0;
  for(auto & col : colls_)
  {
    for(const auto & o : col.objects_)
    {
      // FIXME A bit wasteful if the same convex is implied in multiple collisions
      tvm::utils::transform(*o.o_, o.X_f_o_ * o.f_->position());
    }

    double dist = tvm::utils::distance(*col.pair_, closestPoints_[0], closestPoints_[1]);
    dist = dist >= 0 ? std::sqrt(dist) : -std::sqrt(-dist);
    col.normVecDist_ = (closestPoints_[0] - closestPoints_[1]) / dist;

    for(size_t i = 0; i < col.objects_.size(); ++i)
    {
      auto & o = col.objects_[i];
      o.nearestPoint_ = (sva::PTransformd(closestPoints_[i])*o.f_->position().inv()).translation();
      o.jac_.point(o.nearestPoint_);
    }

    value_(i++) = dist;
  }

}

void CollisionFunction::updateJacobian()
{
  for(int i = 0; i < variables_.numberOfVariables(); ++i)
  {
    jacobian_[variables_[i].get()].setZero();
  }
  Eigen::DenseIndex i = 0;
  for(auto & col : colls_)
  {
    double sign = 1.;
    for(auto & o : col.objects_)
    {
      const auto & r = o.f_->robot();
      const Eigen::MatrixXd & jac = o.jac_.jacobian(r.mb(), r.mbc());
      distJac_.block(0, 0, 1, o.jac_.dof()).noalias() =
        (sign *  col.normVecDist_).transpose() * jac.block(3, 0, 3, o.jac_.dof());
      o.jac_.fullJacobian(r.mb(),
                          distJac_.block(0, 0, 1, o.jac_.dof()),
                          fullJac_);
      int ffSize = r.qFreeFlyer()->space().tSize();
      int jSize = r.qJoints()->space().tSize();
      if(ffSize)
      {
        jacobian_[r.qFreeFlyer().get()].block(i, 0, 1, ffSize) +=
          fullJac_.block(0, 0, 1, ffSize);
      }
      if(jSize)
      {
        jacobian_[r.qJoints().get()].block(i, 0, 1, jSize) +=
          fullJac_.block(0, ffSize, 1, jSize);
      }
      sign *= -1.;
    }
    i++;
  }
}

void CollisionFunction::updateVelocity()
{
  Eigen::DenseIndex i = 0;
  for(auto & col : colls_)
  {
    velocity_(i) = 0.;
    double sign = 1.;
    for(auto & o : col.objects_)
    {
      const auto & r = o.f_->robot();
      Eigen::Vector3d pSpeed = o.jac_.velocity(r.mb(), r.mbc()).linear();
      velocity_(i) += sign * pSpeed.dot(col.normVecDist_);
      sign *= -1.;
    }
    i++;
  }
}

void CollisionFunction::updateNormalAcceleration()
{
  Eigen::DenseIndex i = 0;
  for(auto & col : colls_)
  {
    Eigen::Vector3d speedVec = (col.normVecDist_ - col.prevNormVecDist_) / dt_;
    col.prevNormVecDist_ = col.normVecDist_;
    normalAcceleration_(i) = 0.;
    double sign = 1.;
    for(auto & o : col.objects_)
    {
      const auto & r = o.f_->robot();
      Eigen::Vector3d pNormalAcc = o.jac_.normalAcceleration(r.mb(), r.mbc(), r.normalAccB()).linear();
      Eigen::Vector3d pSpeed = o.jac_.velocity(r.mb(), r.mbc()).linear();
      normalAcceleration_(0) += sign * ( pNormalAcc.dot(col.normVecDist_) + pSpeed.dot(speedVec) );
      sign *= -1.;
    }
  }
}

void CollisionFunction::reset()
{
  colls_.resize(0);
  resize(0);
}

} // namespace robot

} // namespace tvm
