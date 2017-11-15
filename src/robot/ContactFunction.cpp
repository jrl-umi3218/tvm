#include <tvm/robot/internal/ContactFunction.h>

#include <tvm/Robot.h>

namespace tvm
{

namespace robot
{

namespace internal
{

ContactFunction::ContactFunction(ContactPtr contact, Eigen::Matrix6d dof)
: function::abstract::Function(6),
  contact_(contact), dof_(dof)
{
  const auto & f1 = contact_->f1();
  const auto & f2 = contact_->f2();
  const auto & r1 = f1.robot();
  const auto & r2 = f2.robot();
  assert(r1.mb().nrDof() > 0 || r2.mb().nrDof() > 0);

  registerUpdates(
                  Update::Value, &ContactFunction::updateValue,
                  Update::Jacobian, &ContactFunction::updateJacobian,
                  Update::Velocity, &ContactFunction::updateVelocity,
                  Update::NormalAcceleration, &ContactFunction::updateNormalAcceleration
                 );

  addOutputDependency<ContactFunction>(FirstOrderProvider::Output::Value, Update::Value);
  addOutputDependency<ContactFunction>(Output::Velocity, Update::Velocity);
  addOutputDependency<ContactFunction>(Output::NormalAcceleration, Update::NormalAcceleration);
  addOutputDependency<ContactFunction>(FirstOrderProvider::Output::Jacobian, Update::Jacobian);

  has_f1_ = r1.mb().nrDof() > 0;
  has_f2_ = r2.mb().nrDof() > 0;

  // FIXME We must have X_f1_f2. So, at the first iteration, we need X_0_f1
  // and X_0_f2 to be up-to-date at the start
  addInputDependency<ContactFunction>(Update::Value, contact_, Contact::Output::F1Position);
  addInputDependency<ContactFunction>(Update::Value, contact_, Contact::Output::F2Position);

  // Add more dependencies to f1 if needed
  if(has_f1_)
  {
    addInputDependency<ContactFunction>(Update::Velocity, contact_, Contact::Output::F1Velocity);
    addInputDependency<ContactFunction>(Update::NormalAcceleration, contact_, Contact::Output::F1NormalAcceleration);
    addInputDependency<ContactFunction>(Update::Jacobian, contact_, Contact::Output::F1Jacobian);
    addVariable(r1.q(), false);
    jacobian_getter_[r1.q().get()] = [contact]()
    {
      return contact->f1().jacobian();
    };
  }
  // Add more dependencies to f2 if needed
  if(has_f2_)
  {
    addInputDependency<ContactFunction>(Update::Velocity, contact_, Contact::Output::F2Velocity);
    addInputDependency<ContactFunction>(Update::NormalAcceleration, contact_, Contact::Output::F2NormalAcceleration);
    addInputDependency<ContactFunction>(Update::Jacobian, contact_, Contact::Output::F2Jacobian);
    addVariable(r2.q(), false);
    jacobian_getter_[r2.q().get()] = [contact]()
    {
      return contact->f1().jacobian();
    };
  }
}

void ContactFunction::updateValue()
{
  const auto & f1 = contact_->f1();
  const auto & f2 = contact_->f2();
  if(first_update_)
  {
    first_update_ = false;
    // X_f1_f2 = X_0_f2 * X_0_f1.inv():
    X_f1_f2_init_ = f2.position() * f1.position().inv();
  }
  auto X_f1_f2 = f2.position() * f1.position().inv();
  value_ = dof_ * sva::transformError(X_f1_f2, X_f1_f2_init_).vector();
}

void ContactFunction::updateVelocity()
{
  if(has_f1_)
  {
    velocity_ += dof_ * contact_->f1().velocity().vector();
  }
  if(has_f2_)
  {
    velocity_ -= dof_ * contact_->f2().velocity().vector();
  }
}

void ContactFunction::updateNormalAcceleration()
{
  if(has_f1_)
  {
    normalAcceleration_ += dof_ * contact_->f1().normalAcceleration().vector();
  }
  if(has_f2_)
  {
    normalAcceleration_ -= dof_ * contact_->f2().normalAcceleration().vector();
  }
}

void ContactFunction::updateJacobian()
{
  if(has_f1_)
  {
    const auto & f1 = contact_->f1();
    const auto & r1 = f1.robot();
    jacobian_[r1.q().get()] = dof_ * f1.jacobian();
  }
  if(has_f2_)
  {
    const auto & f2 = contact_->f2();
    const auto & r2 = f2.robot();
    jacobian_[r2.q().get()] = dof_ * f2.jacobian();
  }
}

}

}

}
