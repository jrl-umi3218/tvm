#include <tvm/robot/CollisionFunction.h>

#include <tvm/Robot.h>
#include <tvm/utils/sch.h>

namespace tvm
{

namespace robot
{

CollisionFunction::CollisionFunction(Clock & clock)
: function::abstract::Function(0),
  clock_(clock), last_tick_(clock.ticks())
{
  registerUpdates(Update::Value, &CollisionFunction::updateValue,
                  Update::Velocity, &CollisionFunction::updateVelocity,
                  Update::Jacobian, &CollisionFunction::updateJacobian,
                  Update::Time, &CollisionFunction::updateTimeDependency,
                  Update::NormalAcceleration, &CollisionFunction::updateNormalAcceleration);

  addOutputDependency<CollisionFunction>(Output::Value, Update::Value);
  addOutputDependency<CollisionFunction>(Output::Velocity, Update::Velocity);
  addOutputDependency<CollisionFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<CollisionFunction>(Output::NormalAcceleration, Update::NormalAcceleration);

  addInternalDependency<CollisionFunction>(Update::Jacobian, Update::Value);
  addInternalDependency<CollisionFunction>(Update::Velocity, Update::Jacobian);
  addInternalDependency<CollisionFunction>(Update::Time, Update::Value);
  addInternalDependency<CollisionFunction>(Update::NormalAcceleration, Update::Time);
}

void CollisionFunction::addCollision(ConvexHullPtr ch1, ConvexHullPtr ch2)
{
  const auto & r1 = ch1->frame().robot();
  const auto & r2 = ch2->frame().robot();
  int r1Dof = r1.mb().nrDof();
  int r2Dof = r2.mb().nrDof();
  if(r2Dof > r1Dof)
  {
    addCollision(ch2, ch1);
    return;
  }
  assert(r1Dof >= r2Dof);
  colls_.emplace_back(*this, ch1, ch2);
  resize(static_cast<int>(colls_.size()));
  auto maxDof = std::max(fullJac_.cols(), static_cast<Eigen::DenseIndex>(r1Dof));
  fullJac_.resize(1, maxDof);
  distJac_.resize(1, maxDof);
}

CollisionFunction::CollisionData::CollisionData()
: pair_(nullptr, nullptr)
{
}

CollisionFunction::CollisionData::CollisionData(CollisionFunction & fn,
                                                ConvexHullPtr ch1, ConvexHullPtr ch2)
: ch_{ch1, ch2},
  pair_(ch_[0]->makePair(*ch_[1]))
{
  for(size_t i = 0; i < 2; ++i)
  {
    auto & r = ch_[i]->frame().robot();
    int rDof = r.mb().nrDof();
    if(rDof > 0)
    {
      fn.addInputDependency<CollisionFunction>(Update::Value, *ch_[i], ConvexHull::Output::Position);
      fn.addInputDependency<CollisionFunction>(Update::Jacobian, r, Robot::Output::Dynamics);
      fn.addInputDependency<CollisionFunction>(Update::NormalAcceleration, r, Robot::Output::Acceleration);
      objects_.push_back({Eigen::Vector3d::Zero(), ch_[i]->frame().rbdJacobian()});
      fn.addVariable(r.q(), false);
    }
  }
}

void CollisionFunction::updateValue()
{
  Eigen::DenseIndex i = 0;
  for(auto & col : colls_)
  {
    double dist = tvm::utils::distance(col.pair_, closestPoints_[0], closestPoints_[1]);
    dist = dist >= 0 ? std::sqrt(dist) : -std::sqrt(-dist);
    if(dist == 0) { dist = std::numeric_limits<double>::min(); }
    col.normVecDist_ = (closestPoints_[0] - closestPoints_[1]) / dist;

    for(size_t j = 0; j < col.objects_.size(); ++j)
    {
      auto & o = col.objects_[j];
      o.nearestPoint_ = (sva::PTransformd(closestPoints_[i])*col.ch_[j]->frame().position().inv()).translation();
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
    for(size_t j = 0; j < col.objects_.size(); ++j)
    {
      auto & o = col.objects_[j];
      const auto & r = col.ch_[j]->frame().robot();
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
    for(size_t j = 0; j < col.objects_.size(); ++j)
    {
      auto & o = col.objects_[j];
      const auto & r = col.ch_[j]->frame().robot();
      Eigen::Vector3d pSpeed = o.jac_.velocity(r.mb(), r.mbc()).linear();
      velocity_(i) += sign * pSpeed.dot(col.normVecDist_);
      sign *= -1.;
    }
    i++;
  }
}

void CollisionFunction::updateTimeDependency()
{
  if(clock_.ticks() != last_tick_)
  {
    last_tick_ = clock_.ticks();
    for(auto & col : colls_)
    {
      col.speedVec_ = (col.normVecDist_ - col.prevNormVecDist_) / clock_.dt();
      col.prevNormVecDist_ = col.normVecDist_;
    }
  }
}

void CollisionFunction::updateNormalAcceleration()
{
  Eigen::DenseIndex i = 0;
  for(auto & col : colls_)
  {
    normalAcceleration_(i) = 0.;
    double sign = 1.;
    for(size_t j = 0; j < col.objects_.size(); ++j)
    {
      auto & o = col.objects_[j];
      const auto & r = col.ch_[j]->frame().robot();
      Eigen::Vector3d pNormalAcc = o.jac_.normalAcceleration(r.mb(), r.mbc(), r.normalAccB()).linear();
      Eigen::Vector3d pSpeed = o.jac_.velocity(r.mb(), r.mbc()).linear();
      normalAcceleration_(i) += sign * ( pNormalAcc.dot(col.normVecDist_) + pSpeed.dot(col.speedVec_) );
      sign *= -1.;
    }
    ++i;
  }
}

void CollisionFunction::reset()
{
  colls_.resize(0);
  resize(0);
}

} // namespace robot

} // namespace tvm
