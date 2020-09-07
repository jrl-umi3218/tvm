/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <tvm/robot/internal/GeometricContactFunction.h>

#include <tvm/Robot.h>

namespace tvm
{

namespace robot
{

namespace internal
{

GeometricContactFunction::GeometricContactFunction(ContactPtr contact, Eigen::Matrix6d dof)
: function::abstract::Function(6), contact_(contact), dof_(dof)
{
  const auto & f1 = contact_->f1();
  const auto & f2 = contact_->f2();
  const auto & r1 = f1.robot();
  const auto & r2 = f2.robot();
  assert(r1.mb().nrDof() > 0 || r2.mb().nrDof() > 0);

  registerUpdates(Update::Value, &GeometricContactFunction::updateValue, Update::Jacobian,
                  &GeometricContactFunction::updateJacobian, Update::Velocity,
                  &GeometricContactFunction::updateVelocity, Update::NormalAcceleration,
                  &GeometricContactFunction::updateNormalAcceleration);

  addOutputDependency<GeometricContactFunction>(Output::Value, Update::Value);
  addOutputDependency<GeometricContactFunction>(Output::Velocity, Update::Velocity);
  addOutputDependency<GeometricContactFunction>(Output::NormalAcceleration, Update::NormalAcceleration);
  addOutputDependency<GeometricContactFunction>(Output::Jacobian, Update::Jacobian);

  has_f1_ = r1.mb().nrDof() > 0;
  has_f2_ = r2.mb().nrDof() > 0;

  // FIXME We must have X_f1_f2. So, at the first iteration, we need X_0_f1
  // and X_0_f2 to be up-to-date at the start
  addInputDependency<GeometricContactFunction>(Update::Value, contact_, Contact::Output::F1Position);
  addInputDependency<GeometricContactFunction>(Update::Value, contact_, Contact::Output::F2Position);

  // Add more dependencies to f1 if needed
  if(has_f1_)
  {
    addInputDependency<GeometricContactFunction>(Update::Velocity, contact_, Contact::Output::F1Velocity);
    addInputDependency<GeometricContactFunction>(Update::NormalAcceleration, contact_,
                                                 Contact::Output::F1NormalAcceleration);
    addInputDependency<GeometricContactFunction>(Update::Jacobian, contact_, Contact::Output::F1Jacobian);
    addVariable(r1.q(), false);
  }
  // Add more dependencies to f2 if needed
  if(has_f2_)
  {
    addInputDependency<GeometricContactFunction>(Update::Velocity, contact_, Contact::Output::F2Velocity);
    addInputDependency<GeometricContactFunction>(Update::NormalAcceleration, contact_,
                                                 Contact::Output::F2NormalAcceleration);
    addInputDependency<GeometricContactFunction>(Update::Jacobian, contact_, Contact::Output::F2Jacobian);
    addVariable(r2.q(), false);
  }
}

void GeometricContactFunction::updateValue()
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

void GeometricContactFunction::updateVelocity()
{
  velocity_.setZero();
  if(has_f1_)
  {
    velocity_ += dof_ * contact_->f1().velocity().vector();
  }
  if(has_f2_)
  {
    velocity_ -= dof_ * contact_->f2().velocity().vector();
  }
}

void GeometricContactFunction::updateNormalAcceleration()
{
  normalAcceleration_.setZero();
  if(has_f1_)
  {
    normalAcceleration_ += dof_ * contact_->f1().normalAcceleration().vector();
  }
  if(has_f2_)
  {
    normalAcceleration_ -= dof_ * contact_->f2().normalAcceleration().vector();
  }
}

void GeometricContactFunction::updateJacobian()
{
  if(has_f1_)
  {
    const auto & f1 = contact_->f1();
    const auto & r1 = f1.robot();
    splitJacobian(dof_ * f1.jacobian(), r1.q());
  }
  if(has_f2_)
  {
    const auto & f2 = contact_->f2();
    const auto & r2 = f2.robot();
    splitJacobian(-dof_ * f2.jacobian(), r2.q());
  }
}

} // namespace internal

} // namespace robot

} // namespace tvm
