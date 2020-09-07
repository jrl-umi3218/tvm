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

#include <tvm/robot/CoMFunction.h>

namespace tvm
{

namespace robot
{

CoMFunction::CoMFunction(RobotPtr robot) : function::abstract::Function(3), robot_(robot), jac_(robot_->mb())
{
  reset();
  // clang-format off
  registerUpdates(Update::Value, &CoMFunction::updateValue,
                  Update::Velocity, &CoMFunction::updateVelocity,
                  Update::Jacobian, &CoMFunction::updateJacobian,
                  Update::NormalAcceleration, &CoMFunction::updateNormalAcceleration,
                  Update::JDot, &CoMFunction::updateJDot);
  // clang-format on
  addOutputDependency<CoMFunction>(Output::Value, Update::Value);
  addOutputDependency<CoMFunction>(Output::Velocity, Update::Velocity);
  addOutputDependency<CoMFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<CoMFunction>(Output::NormalAcceleration, Update::NormalAcceleration);
  addOutputDependency<CoMFunction>(Output::JDot, Update::JDot);
  addVariable(robot_->q(), false);
  addInputDependency<CoMFunction>(Update::Value, robot_, Robot::Output::CoM);
  addInputDependency<CoMFunction>(Update::Velocity, robot_, Robot::Output::FV);
  addInputDependency<CoMFunction>(Update::Jacobian, robot_, Robot::Output::FV);
  addInputDependency<CoMFunction>(Update::NormalAcceleration, robot_, Robot::Output::NormalAcceleration);
  addInputDependency<CoMFunction>(Update::JDot, robot_, Robot::Output::NormalAcceleration);
}

void CoMFunction::reset() { com_ = robot_->com(); }

void CoMFunction::updateValue() { value_ = robot_->com() - com_; }

void CoMFunction::updateVelocity() { velocity_ = jac_.velocity(robot_->mb(), robot_->mbc()); }

void CoMFunction::updateJacobian() { splitJacobian(jac_.jacobian(robot_->mb(), robot_->mbc()), robot_->q()); }

void CoMFunction::updateNormalAcceleration()
{
  normalAcceleration_ = jac_.normalAcceleration(robot_->mb(), robot_->mbc(), robot_->normalAccB());
}

void CoMFunction::updateJDot() { splitJacobian(jac_.jacobianDot(robot_->mb(), robot_->mbc()), robot_->q()); }

} // namespace robot

} // namespace tvm
