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

#include <tvm/robot/Frame.h>

#include <tvm/Robot.h>

namespace
{
  Eigen::Matrix3d hat(const Eigen::Vector3d & v)
  {
    Eigen::Matrix3d ret;
    ret << 0., -v.z(), v.y(),
           v.z(), 0., -v.x(),
           -v.y(), v.x(), 0.;
    return ret;
  }
}

namespace tvm
{

namespace robot
{

Frame::Frame(std::string name,
             RobotPtr robot,
             const std::string & body,
             sva::PTransformd X_b_f)
: name_(std::move(name)),
  robot_(robot),
  bodyId_(robot->mb().bodyIndexByName(body)),
  jac_(robot->mb(), body),
  X_b_f_(std::move(X_b_f)),
  jacTmp_(6, jac_.dof()),
  jacobian_(6, robot->mb().nrDof()) // FIXME Don't allocate until needed?
{
  registerUpdates(
                  Update::Position, &Frame::updatePosition,
                  Update::Jacobian, &Frame::updateJacobian,
                  Update::Velocity, &Frame::updateVelocity,
                  Update::NormalAcceleration, &Frame::updateNormalAcceleration
                 );


  addOutputDependency(Output::Position, Update::Position);
  addInputDependency(Update::Position, robot_, Robot::Output::FK);

  addOutputDependency(Output::Jacobian, Update::Jacobian);
  addInternalDependency(Update::Jacobian, Update::Position);
  addInputDependency(Update::Jacobian, robot_, Robot::Output::FV);

  addOutputDependency(Output::Velocity, Update::Velocity);
  addInputDependency(Update::Velocity, robot_, Robot::Output::FV);

  addOutputDependency(Output::NormalAcceleration, Update::NormalAcceleration);
  addInputDependency(Update::NormalAcceleration, robot_, Robot::Output::NormalAcceleration);

  /** Initialize all data */
  updatePosition();
  updateJacobian();
  updateVelocity();
  updateNormalAcceleration();
}

void Frame::updatePosition()
{
  const auto & X_0_b = robot_->mbc().bodyPosW[bodyId_];
  /** X_0_f */
  position_ = X_b_f_ * X_0_b;
}

void Frame::updateJacobian()
{
  assert(jacobian_.rows() == 6 && jacobian_.cols() == robot_->mb().nrDof());
  const auto & partialJac = jac_.jacobian(robot_->mb(),
                                          robot_->mbc());
  jacTmp_ = partialJac;
  Eigen::Matrix3d h = -hat(robot_->mbc().bodyPosW[bodyId_].rotation().transpose() * X_b_f_.translation());
  jacTmp_.block(3, 0, 3, jac_.dof()).noalias() += h * partialJac.block(3, 0, 3, jac_.dof());
  jac_.fullJacobian(robot_->mb(), jacTmp_, jacobian_);
}

void Frame::updateVelocity()
{
  velocity_ = X_b_f_ * robot_->mbc().bodyVelW[bodyId_];
}

void Frame::updateNormalAcceleration()
{
  normalAcceleration_ = X_b_f_ * jac_.normalAcceleration(robot_->mb(), robot_->mbc(), robot_->normalAccB());
}

const std::string & Frame::body() const
{
  return robot_->mb().body(bodyId_).name();
}

}

}
