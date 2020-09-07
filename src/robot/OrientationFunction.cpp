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

#include <tvm/Robot.h>
#include <tvm/robot/OrientationFunction.h>

namespace tvm
{

namespace robot
{

OrientationFunction::OrientationFunction(FramePtr frame) : function::abstract::Function(3), frame_(frame)
{
  reset();
  // clang-format off
  registerUpdates(Update::Value, &OrientationFunction::updateValue,
                  Update::Velocity, &OrientationFunction::updateVelocity,
                  Update::Jacobian, &OrientationFunction::updateJacobian,
                  Update::NormalAcceleration, &OrientationFunction::updateNormalAcceleration);
  // clang-format on
  addOutputDependency<OrientationFunction>(Output::Value, Update::Value);
  addOutputDependency<OrientationFunction>(Output::Velocity, Update::Velocity);
  addOutputDependency<OrientationFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<OrientationFunction>(Output::NormalAcceleration, Update::NormalAcceleration);
  const auto & robot = frame_->robot();
  addVariable(robot.q(), false);
  addInputDependency<OrientationFunction>(Update::Value, frame_, Frame::Output::Position);
  addInputDependency<OrientationFunction>(Update::Velocity, frame_, Frame::Output::Velocity);
  addInputDependency<OrientationFunction>(Update::Jacobian, frame_, Frame::Output::Jacobian);
  addInputDependency<OrientationFunction>(Update::NormalAcceleration, frame_, Frame::Output::NormalAcceleration);
}

void OrientationFunction::reset() { ori_ = frame_->position().rotation(); }

void OrientationFunction::updateValue() { value_ = sva::rotationError(ori_, frame_->position().rotation()); }

void OrientationFunction::updateVelocity() { velocity_ = frame_->velocity().angular(); }

void OrientationFunction::updateJacobian() { splitJacobian(frame_->jacobian().topRows<3>(), frame_->robot().q()); }

void OrientationFunction::updateNormalAcceleration() { normalAcceleration_ = frame_->normalAcceleration().angular(); }

} // namespace robot

} // namespace tvm
