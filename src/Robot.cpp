#include <tvm/Robot.h>

#include <tvm/Space.h>

#include <RBDyn/EulerIntegration.h>
#include <RBDyn/FK.h>
#include <RBDyn/FV.h>
#include <RBDyn/FA.h>

namespace tvm
{

Robot::Robot(const std::string & name, rbd::MultiBody mb, rbd::MultiBodyConfig mbc)
: name_(name), mb_(mb), mbc_(mbc),
  normalAccB_(mbc_.bodyAccB.size()), fd_(mb_),
  q_(tvm::Space(mb_.nrDof(), mb_.nrParams(), mb_.nrDof()).createVariable("q")),
  tau_(tvm::Space(mb_.nrDof()).createVariable("tau"))
{
  updateTimeDependency(0.);
}

void Robot::updateTimeDependency(double dt)
{
  rbd::eulerIntegration(mb_, mbc_, dt);
  rbd::forwardKinematics(mb_, mbc_);
  rbd::forwardVelocity(mb_, mbc_);
  rbd::forwardAcceleration(mb_, mbc_);
  computeNormalAccB();
}

void Robot::computeNormalAccB()
{
  // No need to compute that if the robot is not actuated
  if(mb_.nrDof() > 0)
  {
    const auto & pred = mb_.predecessors();
    const auto & succ = mb_.successors();
    for(int i = 0; i < mb_.nrJoints(); ++i)
    {
      const auto & X_p_i = mbc_.parentToSon[i];
      const auto & vj_i = mbc_.jointVelocity[i];
      const auto & vb_i = mbc_.bodyVelB[i];
      if(pred[i] != -1)
      {
        normalAccB_[succ[i]] = X_p_i * normalAccB_[pred[i]] + vb_i.cross(vj_i);
      }
      else
      {
        normalAccB_[succ[i]] = vb_i.cross(vj_i);
      }
    }
  }
}

}
