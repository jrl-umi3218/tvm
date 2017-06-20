#include "Mockup.h"
#include <string>

RobotMockup::RobotMockup()
  : DataNode({ K1, K2, K3, V1, V2, D1, D2, A1 }, { Kinematics, Velocity, Dynamics, Acceleration })
  , k(0), v(0), d(0), a(0)
{
}

void RobotMockup::update_(const taskvm::internal::UnifiedEnumValue& u)
{
  switch (Update(u))
  {
  case Kinematics: updateK(); break;
  case Velocity: updateV(); break;
  case Dynamics: updateD(); break;
  case Acceleration: updateA(); break;
  default: break;
  }
}

void RobotMockup::fillOutputDependencies()
{
  addOutputDependency({ K1,K2,K3 }, Kinematics);
  addOutputDependency({ V1,V2 }, Velocity);
  addOutputDependency({ D1,D2 }, Dynamics);
  addOutputDependency(A1, { Acceleration, Dynamics, Velocity, Kinematics });
}

void RobotMockup::fillInternalDependencies()
{
  addInternalDependency(Velocity, Kinematics);
  addInternalDependency(Dynamics, Velocity);
  addInternalDependency(Acceleration, Velocity);
}

void RobotMockup::fillUpdateDependencies()
{
  //do nothing
}

FunctionMockup::FunctionMockup(std::initializer_list<Output> outputs, std::initializer_list<Update> updates)
  :DataNode(outputs, updates)
{
}

RobotFunction::RobotFunction(std::initializer_list<Output> outputs, std::shared_ptr<RobotMockup> robot)
  : FunctionMockup(outputs, {Update::Value, Update::Velocity, Update::JDot})
  , robot_(robot)
{
  addInput(robot, { RobotMockup::K1, RobotMockup::K2, RobotMockup::K3, RobotMockup::V1, RobotMockup::V2, RobotMockup::D1, RobotMockup::D2 });
}

void RobotFunction::fillInternalDependencies()
{
  addInternalDependency(Update::Velocity, Update::Value);
  addInternalDependency(Update::JDot, Update::Velocity);
}

void RobotFunction::fillUpdateDependencies()
{
  addInputDependency(Update::Value, *robot_, RobotMockup::K1);
  addInputDependency(Update::Value, *robot_, RobotMockup::K2);
  addInputDependency(Update::Value, *robot_, RobotMockup::K3);
  addInputDependency(Update::Velocity, *robot_, RobotMockup::V1);
  addInputDependency(Update::Velocity, *robot_, RobotMockup::V2);
  addInputDependency(Update::JDot, *robot_, RobotMockup::V1);
  addInputDependency(Update::JDot, *robot_, RobotMockup::V2);
}

SomeRobotFunction1::SomeRobotFunction1(std::shared_ptr<RobotMockup> robot)
  :RobotFunction({Output::Value, Output::Jacobian, Output::Velocity, Output::NormalAcceleration, Output::JDot }, robot)
{
}

void SomeRobotFunction1::update_(const taskvm::internal::UnifiedEnumValue & u)
{
  switch (Update(u))
  {
  case Update::Value: 
    std::cout << "update SomeRobotFunction1::Value" << std::endl; 
    val = (int)robot_->getK1() + (int)robot_->getK2();
    break;
  case Update::Velocity: 
    std::cout << "update SomeRobotFunction1::Velocity" << std::endl; 
    j = (int)robot_->getV1();
    vel = (int)robot_->getV2();
    na = (int)robot_->getK3()*(int)robot_->getV1();
    break;
  case Update::JDot: 
    std::cout << "update SomeRobotFunction1::JDot" << std::endl; 
    jdot = (int)robot_->getK3()*(int)robot_->getV1() + (int)robot_->getV2();
    break;
  default: break;
  }
}

void SomeRobotFunction1::fillOutputDependencies()
{
  addOutputDependency(Output::Value, Update::Value);
  addOutputDependency(Output::Jacobian, Update::Velocity);
  addOutputDependency(Output::Velocity, Update::Velocity);
  addOutputDependency(Output::NormalAcceleration, Update::Velocity);
  addOutputDependency(Output::JDot, Update::JDot);
}

SomeRobotFunction2::SomeRobotFunction2(std::shared_ptr<RobotMockup> robot)
  :RobotFunction({ Output::Value, Output::Jacobian }, robot)
{
}

void SomeRobotFunction2::update_(const taskvm::internal::UnifiedEnumValue & u)
{
  switch (Update(u))
  {
  case Update::Value: 
    val = robot_->getK1();
    std::cout << "update SomeRobotFunction2::Value" << std::endl; 
    break;
  case Update::Velocity:
    j = robot_->getV1();
    std::cout << "update SomeRobotFunction2::Velocity" << std::endl; 
    break;
  default: break;
  }
}

void SomeRobotFunction2::fillOutputDependencies()
{
  addOutputDependency(Output::Value, Update::Value);
  addOutputDependency(Output::Jacobian, Update::Velocity);
}


LinearConstraint::LinearConstraint(const std::string& name)
  : DataNode({Output::Value, Output::A, Output::b}, {Update::Matrices})
  , name_(name)
{
}

Dummy<int(LinearConstraint::Output::Value)> LinearConstraint::value(int x) const
{
  //obviously, in real setting we would depend from a variable and take its value
  return A_*x + b_;
}

Dummy<int(LinearConstraint::Output::A)> LinearConstraint::A() const
{
  return A_;
}

Dummy<int(LinearConstraint::Output::b)> LinearConstraint::b() const
{
  return b_;
}

void LinearConstraint::update_(const taskvm::internal::UnifiedEnumValue& u)
{
  switch (Update(u))
  {
  case Update::Matrices:
    std::cout << "update LinearConstraint(" << name_ << ")::Matrices" << std::endl;
    updateMatrices();
    break;
  default:
    break;
  }
}

void LinearConstraint::fillOutputDependencies()
{
  addOutputDependency({ Output::Value, Output::A, Output::b }, Update::Matrices);
}

void LinearConstraint::fillInternalDependencies()
{
  //do nothing
}



KinematicLinearizedConstraint::KinematicLinearizedConstraint(const std::string& name, std::shared_ptr<FunctionMockup> function)
  : LinearConstraint(name)
  , function_(function)
{
  addInput(function, { FunctionMockup::Output::Value, FunctionMockup::Output::Jacobian });
}

void KinematicLinearizedConstraint::fillUpdateDependencies()
{
  addInputDependency(Update::Matrices, *function_, FunctionMockup::Output::Value);
  addInputDependency(Update::Matrices, *function_, FunctionMockup::Output::Jacobian);
}

void KinematicLinearizedConstraint::updateMatrices()
{
  A_ = function_->jacobian();
  b_ = function_->value();
}

DynamicLinearizedConstraint::DynamicLinearizedConstraint(const std::string& name, std::shared_ptr<FunctionMockup> function)
  : LinearConstraint(name)
  , function_(function)
{
  addInput(function, { FunctionMockup::Output::Value, 
                       FunctionMockup::Output::Jacobian,
                       FunctionMockup::Output::Velocity, 
                       FunctionMockup::Output::NormalAcceleration });
}

void DynamicLinearizedConstraint::fillUpdateDependencies()
{
  addInputDependency(Update::Matrices, *function_, FunctionMockup::Output::Value);
  addInputDependency(Update::Matrices, *function_, FunctionMockup::Output::Jacobian);
  addInputDependency(Update::Matrices, *function_, FunctionMockup::Output::Velocity);
  addInputDependency(Update::Matrices, *function_, FunctionMockup::Output::NormalAcceleration);
}

void DynamicLinearizedConstraint::updateMatrices()
{
  A_ = function_->jacobian();
  b_ = function_->value() + function_->velocity() + function_->normalAcc();
}

DynamicEquation::DynamicEquation(const std::string& name, std::shared_ptr<RobotMockup> robot)
  : LinearConstraint(name)
  , robot_(robot)
{
  addInput(robot, { RobotMockup::D1, RobotMockup::D2 });
}

void DynamicEquation::fillUpdateDependencies()
{
  addInputDependency(Update::Matrices, *robot_, RobotMockup::D1);
  addInputDependency(Update::Matrices, *robot_, RobotMockup::D2);
}

void DynamicEquation::updateMatrices()
{
  A_ = robot_->getD1();
  b_ = robot_->getD2();
}
