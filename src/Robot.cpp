#include <tvm/Robot.h>

#include <tvm/Space.h>

#include <RBDyn/CoM.h>
#include <RBDyn/EulerIntegration.h>
#include <RBDyn/FK.h>
#include <RBDyn/FV.h>
#include <RBDyn/FA.h>

namespace tvm
{

Robot::Robot(const std::string & name, rbd::MultiBodyGraph & mbg, rbd::MultiBody mb, rbd::MultiBodyConfig mbc)
: name_(name), mb_(mb), mbc_(mbc),
  normalAccB_(mbc_.bodyAccB.size()), fd_(mb_),
  bodyTransforms_(mbg.bodiesBaseTransform(mb_.body(0).name())),
  q_(tvm::Space(mb_.nrDof(), mb_.nrParams(), mb_.nrDof()).createVariable("q")),
  tau_(tvm::Space(mb_.nrDof()).createVariable("tau"))
{
  registerUpdates(Update::q, &Robot::update,
                  Update::CoM, &Robot::updateCoM,
                  Update::H, &Robot::updateH,
                  Update::C, &Robot::updateC);
  addOutputDependency(Output::q, Update::q);
  addInternalDependency(Update::CoM, Update::q);
  addInternalDependency(Update::H, Update::q);
  addInternalDependency(Update::C, Update::q);
  addOutputDependency(Output::CoM, Update::CoM);
  addOutputDependency(Output::H, Update::H);
  addOutputDependency(Output::C, Update::C);
  update();
}

void Robot::updateTimeDependency(double dt)
{
  auto ddq = dot(q_, 2)->value();
  rbd::vectorToParam(ddq, mbc_.alphaD);
  rbd::eulerIntegration(mb_, mbc_, dt);
  auto dq = dot(q_, 1)->value();
  rbd::paramToVector(mbc_.alpha, dq);
  dot(q_, 1)->value(dq);
  auto q = q_->value();
  rbd::paramToVector(mbc_.q, q);
  q_->value(q);
}

void Robot::update()
{
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

void Robot::updateH()
{
  fd_.computeH(mb_, mbc_);
}

void Robot::updateC()
{
  fd_.computeC(mb_, mbc_);
}

void Robot::updateCoM()
{
  com_ = rbd::computeCoM(mb_, mbc_);
}

}
