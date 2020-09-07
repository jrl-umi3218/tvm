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

#include <tvm/robot/PostureFunction.h>

namespace tvm
{

namespace robot
{

PostureFunction::PostureFunction(RobotPtr robot)
: function::abstract::Function(robot->qJoints()->space().tSize()), robot_(robot),
  j0_(robot_->mb().joint(0).type() == rbd::Joint::Free ? 1 : 0)
{
  // clang-format off
  registerUpdates(Update::Value, &PostureFunction::updateValue,
                  Update::Velocity, &PostureFunction::updateVelocity);
  // clang-format on
  addOutputDependency<PostureFunction>(Output::Value, Update::Value);
  addOutputDependency<PostureFunction>(Output::Velocity, Update::Velocity);
  addInputDependency<PostureFunction>(Update::Value, robot_, Robot::Output::FK);
  addInputDependency<PostureFunction>(Update::Velocity, robot_, Robot::Output::FK);
  addVariable(robot->qJoints(), false);
  jacobian_[robot->qJoints().get()].setIdentity();
  jacobian_[robot->qJoints().get()].properties({tvm::internal::MatrixProperties::IDENTITY});
  JDot_[robot->qJoints().get()].setZero();
  normalAcceleration_.setZero();
  value_.setZero();
  velocity_.setZero();
  reset();
}

void PostureFunction::reset() { posture_ = robot_->mbc().q; }

void PostureFunction::posture(const std::string & j, const std::vector<double> & q)
{
  // This assert throws if j is not part of robot.mb
  assert(robot_->mb().sJointIndexByName(j));
  auto jIndex = robot_->mb().jointIndexByName(j);
  assert(posture_[jIndex].size() == q.size());
  posture_[jIndex] = q;
}

namespace
{
bool isValidPosture(const std::vector<std::vector<double>> & ref, const std::vector<std::vector<double>> & in)
{
  if(ref.size() != in.size())
  {
    return false;
  }
  for(size_t i = 0; i < ref.size(); ++i)
  {
    if(ref[i].size() != in[i].size())
    {
      return false;
    }
  }
  return true;
}
} // namespace

void PostureFunction::posture(const std::vector<std::vector<double>> & p)
{
  assert(isValidPosture(posture_, p));
  posture_ = p;
}

void PostureFunction::updateValue()
{
  int pos = 0;
  for(int jI = j0_; jI < robot_->mb().nrJoints(); ++jI)
  {
    const auto & j = robot_->mb().joint(jI);
    if(j.dof() == 1) // prismatic or revolute
    {
      value_(pos) = robot_->mbc().q[jI][0] - posture_[jI][0];
      pos++;
    }
    else if(j.dof() == 4) // spherical
    {
      Eigen::Matrix3d ori(
          Eigen::Quaterniond(posture_[jI][0], posture_[jI][1], posture_[jI][2], posture_[jI][3]).matrix());
      auto error = sva::rotationError(ori, robot_->mbc().jointConfig[jI].rotation());
      value_.segment(pos, 3) = error;
      pos += 3;
    }
  }
}

void PostureFunction::updateVelocity()
{
  int pos = 0;
  for(int jI = j0_; jI < robot_->mb().nrJoints(); ++jI)
  {
    for(auto & qI : robot_->mbc().alpha[jI])
    {
      velocity_(pos) = qI;
      pos++;
    }
  }
}

} // namespace robot

} // namespace tvm
