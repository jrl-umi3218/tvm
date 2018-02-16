#include <tvm/Robot.h>

#include <tvm/Space.h>

#include <RBDyn/CoM.h>
#include <RBDyn/EulerIntegration.h>
#include <RBDyn/FK.h>
#include <RBDyn/FV.h>
#include <RBDyn/FA.h>

namespace tvm
{

Robot::Robot(Clock & clock, const std::string & name, rbd::MultiBodyGraph & mbg, rbd::MultiBody mb, rbd::MultiBodyConfig mbc)
: clock_(clock), name_(name), mb_(mb), mbc_(mbc),
  normalAccB_(mbc_.bodyAccB.size()), fd_(mb_),
  bodyTransforms_(mbg.bodiesBaseTransform(mb_.body(0).name())),
  tau_(tvm::Space(mb_.nrDof()).createVariable("tau"))
{
  if(mb.nrJoints() > 0 && mb.joint(0).type() == rbd::Joint::Free)
  {
    q_ff_ = tvm::Space(6, 7, 6).createVariable("qFreeFlyer");
    q_joints_ = tvm::Space(mb.nrDof() - 6, mb.nrParams() - 7, mb.nrDof() - 6).createVariable("qJoints");
  }
  else
  {
    q_ff_ = tvm::Space(0).createVariable("qFreeFlyer");
    q_joints_ = tvm::Space(mb.nrDof(), mb.nrParams(), mb.nrDof()).createVariable("qJoints");
  }
  q_.add(q_ff_);
  q_.add(q_joints_);
  dq_ = dot(q_, 1);
  ddq_ = dot(q_, 2);
  registerUpdates(Update::Time, &Robot::updateTimeDependency,
                  Update::Kinematics, &Robot::updateKinematics,
                  Update::Dynamics, &Robot::updateDynamics,
                  Update::Acceleration, &Robot::updateAcceleration,
                  Update::CoM, &Robot::updateCoM,
                  Update::H, &Robot::updateH,
                  Update::C, &Robot::updateC);
  addInputDependency(Update::Time, clock_, Clock::Output::Time);
  addOutputDependency(Output::Kinematics, Update::Kinematics);
  addOutputDependency(Output::Dynamics, Update::Dynamics);
  addOutputDependency(Output::Acceleration, Update::Acceleration);
  addInternalDependency(Update::Kinematics, Update::Time);
  addInternalDependency(Update::CoM, Update::Kinematics);
  addInternalDependency(Update::H, Update::Dynamics);
  addInternalDependency(Update::C, Update::Dynamics);
  addOutputDependency(Output::CoM, Update::CoM);
  addOutputDependency(Output::H, Update::H);
  addOutputDependency(Output::C, Update::C);

  // Compute mass
  mass_ = 0;
  for(const auto & b : mb_.bodies())
  {
    mass_ += b.inertia().mass();
  }

  // Make sure initial robot quantities are well initialized
  updateKinematics();
  updateDynamics();
  updateAcceleration();
  if(mass_ > 0)
  {
    updateCoM();
  }
}

void Robot::updateTimeDependency()
{
  if(last_tick_ != clock_.ticks())
  {
    auto ddq = ddq_.value();
    rbd::vectorToParam(ddq, mbc_.alphaD);
    rbd::eulerIntegration(mb_, mbc_, clock_.dt());
    auto dq = dq_.value();
    rbd::paramToVector(mbc_.alpha, dq);
    dq_.value(dq);
    auto q = q_.value();
    rbd::paramToVector(mbc_.q, q);
    q_.value(q);
    last_tick_ = clock_.ticks();
  }
}

void Robot::updateKinematics()
{
  rbd::forwardKinematics(mb_, mbc_);
}

void Robot::updateDynamics()
{
  rbd::forwardVelocity(mb_, mbc_);
}

void Robot::updateAcceleration()
{
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
