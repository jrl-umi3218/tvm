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

#include <tvm/robot/PositionFunction.h>
#include <tvm/Robot.h>

namespace tvm
{

namespace robot
{

PositionFunction::PositionFunction(FramePtr frame)
: function::abstract::Function(3),
  frame_(frame)
{
  reset();
  registerUpdates(Update::Value, &PositionFunction::updateValue,
                  Update::Velocity, &PositionFunction::updateVelocity,
                  Update::Jacobian, &PositionFunction::updateJacobian,
                  Update::NormalAcceleration, &PositionFunction::updateNormalAcceleration);
  addOutputDependency<PositionFunction>(Output::Value, Update::Value);
  addOutputDependency<PositionFunction>(Output::Velocity, Update::Velocity);
  addOutputDependency<PositionFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<PositionFunction>(Output::NormalAcceleration, Update::NormalAcceleration);
  const auto & robot = frame_->robot();
  addVariable(robot.q(), false);
  addInputDependency<PositionFunction>(Update::Value, frame_, Frame::Output::Position);
  addInputDependency<PositionFunction>(Update::Velocity, frame_, Frame::Output::Velocity);
  addInputDependency<PositionFunction>(Update::Jacobian, frame_, Frame::Output::Jacobian);
  addInputDependency<PositionFunction>(Update::NormalAcceleration, frame_, Frame::Output::NormalAcceleration);
}

void PositionFunction::reset()
{
  pos_ = frame_->position().translation();
}

void PositionFunction::updateValue()
{
  value_ = frame_->position().translation() - pos_;
}

void PositionFunction::updateVelocity()
{
  velocity_ = frame_->velocity().linear();
}

void PositionFunction::updateJacobian()
{
  splitJacobian(frame_->jacobian().bottomRows<3>(), frame_->robot().q());
}

void PositionFunction::updateNormalAcceleration()
{
  normalAcceleration_ = frame_->normalAcceleration().linear();
}

} // namespace robot

} // namespace tvm
