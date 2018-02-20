#include <tvm/robot/internal/DynamicFunction.h>

#include <tvm/robot/Frame.h>

#include <tvm/ControlProblem.h>
#include <tvm/exception/exceptions.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/utils/ProtoTask.h>

#include <iostream>

namespace tvm
{

namespace robot
{

namespace internal
{

DynamicFunction::DynamicFunction(RobotPtr robot)
: function::abstract::LinearFunction(robot->mb().nrDof()),
  robot_(robot)
{
  registerUpdates(Update::Jacobian, &DynamicFunction::updateJacobian);
  addOutputDependency<DynamicFunction>(Output::Jacobian, Update::Jacobian);
  addInputDependency<DynamicFunction>(Update::Jacobian, robot, Robot::Output::H);
  addInputDependency<DynamicFunction>(Update::Value, robot, Robot::Output::C);
  addVariable(dot(robot->q(), 2), true);
  addVariable(robot->tau(), true);
  velocity_.setZero();
  normalAcceleration_.setZero();
}

bool DynamicFunction::addContact(ContactPtr contact, bool linearize,
                                 double mu, unsigned int nrGen)
{
  auto it = getContact(contact->id());
  if(it != contacts_.end()) { return false; }
  bool added = false;
  if(contact->f1().robot().name() == robot_->name())
  {
    added = true;
    addContact_(contact->f1View(), linearize, mu, nrGen, 1);
  }
  if(contact->f2().robot().name() == robot_->name())
  {
    added = true;
    addContact_(contact->f2View(), linearize, mu, nrGen, -1);
  }
  // FIXME When contact forces are being added for two, we may want to return the variables that were added so that we can have opposite forces constraint
  if(added) { std::cout << "Added contact " << contact->id() << std::endl; }
  return added;
}

void DynamicFunction::addContact_(const Contact::View & contact, bool linearize,
                                  double mu, unsigned int nrGen, double dir)
{
  const auto & cPoints = contact.points;
  ForceContact fc;
  fc.id_ = contact.id;
  fc.linearized_ = linearize;
  fc.force_jac_.resize(6, contact.f->rbdJacobian().dof());
  fc.full_jac_.resize(6, contact.f->robot().mb().nrDof());
  FramePtr f = contact.f;
  size_t nrP = cPoints.size();
  if(!linearize)
  {
    for(size_t i = 0; i < nrP; ++i)
    {
      fc.forces_.push_back(tvm::Space(3).createVariable("force"));
      fc.forces_.back()->value(Eigen::Vector3d::Zero());
    }
    fc.updateJacobians_ = [f,cPoints,dir](ForceContact & self,
                              DynamicFunction & df)
    {
      const auto & bodyJac = f->rbdJacobian().bodyJacobian(df.robot_->mb(), df.robot_->mbc());
      for(size_t i = 0; i < self.forces_.size(); ++i)
      {
        const auto & v = self.forces_[i];
        const auto & p = cPoints[i];
        f->rbdJacobian().translateBodyJacobian(bodyJac, df.robot_->mbc(), p.translation(), self.force_jac_);
        f->rbdJacobian().fullJacobian(df.robot_->mb(),
                                      self.force_jac_,
                                      self.full_jac_);
        df.jacobian_[v.get()].noalias() = dir*self.full_jac_.block(3, 0, 3, f->robot().mb().nrDof()).transpose();
      }
    };
    fc.force_ = [cPoints](const ForceContact & self)
    {
      sva::ForceVecd ret {Eigen::Vector6d::Zero()};
      for(size_t i = 0; i < self.forces_.size(); ++i)
      {
        const auto & f = self.forces_[i];
        const auto & p = cPoints[i];
        sva::ForceVecd f_p { Eigen::Vector3d::Zero(),
                             f->value() };
        ret += p.transMul(f_p);
      }
      return ret;
    };
  }
  else
  {
    std::vector<Eigen::Vector3d> generators;
    size_t nrLambda = 0;
    for(const auto & p : cPoints)
    {
      // Note: we could stack the lambdas for each contact point here
      for(size_t j = 0; j < nrGen; ++j)
      {
        std::stringstream ss; ss << f->name() << "_lambda_" << nrLambda++;
        fc.forces_.push_back(tvm::Space(1).createVariable(ss.str()));
        fc.forces_.back()->value(Eigen::VectorXd::Zero(1));
      }
      FrictionCone cone {dir*p.rotation(), nrGen, mu, dir};
      for(auto & g : cone.generators)
      {
        generators.push_back(-g);
      }
    }
    fc.updateJacobians_ = [f, cPoints, generators, nrGen]
      (ForceContact & self, DynamicFunction & df)
    {
      const auto & bodyJac = f->rbdJacobian().bodyJacobian(df.robot_->mb(), df.robot_->mbc());
      for(size_t i = 0; i < cPoints.size(); ++i)
      {
        const auto & p = cPoints[i];
        f->rbdJacobian().translateBodyJacobian(bodyJac, df.robot_->mbc(), p.translation(), self.force_jac_);
        for(size_t j = 0; j < nrGen; ++j)
        {
          const auto & lambda = self.forces_[nrGen*i + j];
          const auto & generator = generators[nrGen*i + j];
          f->rbdJacobian().fullJacobian(df.robot_->mb(),
                                         self.force_jac_,
                                         self.full_jac_);
          df.jacobian_[lambda.get()].noalias() = (generator.transpose() * self.full_jac_.block(3, 0, 3, f->robot().mb().nrDof())).transpose();
        }
      }
    };
    fc.force_ = [cPoints, generators, nrGen](const ForceContact & self)
    {
      sva::ForceVecd ret {Eigen::Vector6d::Zero()};
      for(size_t i = 0; i < cPoints.size(); ++i)
      {
        const auto & p = cPoints[i];
        for(size_t j = 0; j < nrGen; ++j)
        {
          const auto & lambda = self.forces_[nrGen*i + j];
          const auto & generator = -generators[nrGen*i + j];
          sva::ForceVecd f_p { Eigen::Vector3d::Zero(),
                               generator*lambda->value() };
          ret += p.transMul(f_p);
        }
      }
      return ret;
    };
  }
  for(const auto & var : fc.forces_)
  {
    addVariable(var, true);
  }
  addInputDependency<DynamicFunction>(Update::Jacobian, f, Frame::Output::Jacobian);
  contacts_.push_back(fc);
}

void DynamicFunction::removeContact(const Contact::Id & id)
{
  auto it = getContact(id);
  if(it != contacts_.end()) { contacts_.erase(it); }
}

sva::ForceVecd DynamicFunction::contactForce(const Contact::Id & id) const
{
  auto it = getContact(id);
  if(it != contacts_.end())
  {
    return (*it).force_(*it);
  }
  else
  {
    std::cerr << "No contact in the dynamic function" << std::endl;
    return sva::ForceVecd(Eigen::Vector6d::Zero());
  }
}

void DynamicFunction::addPositiveLambdaToProblem(ControlProblem & problem)
{
  for(const auto & c : contacts_)
  {
    if(c.linearized_)
    {
      for(const auto & f : c.forces_)
      {
        auto id = std::make_shared<tvm::function::IdentityFunction>(f);
        problem.add(id >= 0., task_dynamics::None(), {requirements::PriorityLevel{0}});
      }
    }
  }
}

void DynamicFunction::updateValue_()
{
  b_ = robot_->C();
  LinearFunction::updateValue_();
}

void DynamicFunction::updateJacobian()
{
  splitJacobian(robot_->H(), dot(robot_->q(), 2));
  jacobian_[robot_->tau().get()] =  - Eigen::MatrixXd::Identity(robot_->mb().nrDof(), robot_->mb().nrDof());
  for(auto & c : contacts_)
  {
    c.updateJacobians_(c, *this);
  }
}

std::vector<DynamicFunction::ForceContact>::iterator
  DynamicFunction::getContact(const Contact::Id & id)
{
  return std::find_if(contacts_.begin(), contacts_.end(),
                      [&id](const ForceContact & in) { return in.id_ == id; });
}

std::vector<DynamicFunction::ForceContact>::const_iterator
  DynamicFunction::getContact(const Contact::Id & id) const
{
  return std::find_if(contacts_.cbegin(), contacts_.cend(),
                      [&id](const ForceContact & in) { return in.id_ == id; });
}

}

}

}
