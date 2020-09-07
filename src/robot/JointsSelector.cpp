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

#include <tvm/robot/JointsSelector.h>

#include <tvm/exception/exceptions.h>

namespace tvm
{

namespace robot
{

std::unique_ptr<JointsSelector> JointsSelector::ActiveJoints(FunctionPtr f,
                                                             RobotPtr robot,
                                                             const std::vector<std::string> & activeJoints)
{
  auto checkRobotVariable = [&](const std::string & name, const VariablePtr & v) {
    if(v->size())
    {
      const auto & f_vars = f->variables();
      auto it = std::find(f_vars.begin(), f_vars.end(), v);
      if(it == f_vars.end())
      {
        throw exception::FunctionException("Cannot setup joint selector for the provided function: it does not uses "
                                           + name + " variable of " + robot->name());
      }
    }
  };
  checkRobotVariable("free-flyer", robot->qFreeFlyer());
  checkRobotVariable("joints", robot->qJoints());

  const auto & mb = robot->mb();

  std::vector<std::string> joints = activeJoints;
  std::sort(joints.begin(), joints.end(), [&mb](const std::string & lhs, const std::string & rhs) {
    return mb.jointPosInDof(mb.jointIndexByName(lhs)) < mb.jointPosInDof(mb.jointIndexByName(rhs));
  });

  bool ffActive = joints.size() > 0 && joints[0] == mb.joint(0).name() && mb.joint(0).dof() == 6;
  if(ffActive)
  {
    joints.erase(joints.begin());
  }

  Eigen::DenseIndex ffSize = mb.joint(0).dof() == 6 ? 6 : 0;
  std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex>> activeIndex;
  Eigen::DenseIndex start = 0;
  Eigen::DenseIndex size = 0;
  for(const auto & j : joints)
  {
    auto jIndex = mb.jointIndexByName(j);
    auto jDof = mb.joint(jIndex).dof();
    Eigen::DenseIndex pos = mb.jointPosInDof(jIndex) - ffSize;
    if(pos != start + size)
    {
      if(size != 0)
      {
        activeIndex.push_back({start, size});
      }
      start = pos;
      size = jDof;
    }
    else
    {
      size += jDof;
    }
  }
  if(size != 0)
  {
    activeIndex.push_back({start, size});
  }

  return std::unique_ptr<JointsSelector>(new JointsSelector(f, robot, ffActive, activeIndex));
}

std::unique_ptr<JointsSelector> JointsSelector::InactiveJoints(FunctionPtr f,
                                                               RobotPtr robot,
                                                               const std::vector<std::string> & inactiveJoints)
{
  std::vector<std::string> activeJoints{};
  for(const auto & j : robot->mb().joints())
  {
    if(std::find_if(inactiveJoints.begin(), inactiveJoints.end(), [&j](const std::string & s) { return s == j.name(); })
       == inactiveJoints.end())
    {
      activeJoints.push_back(j.name());
    }
  }
  return ActiveJoints(f, robot, activeJoints);
}

JointsSelector::JointsSelector(FunctionPtr f,
                               RobotPtr robot,
                               bool ffActive,
                               const std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex>> & activeIndex)
: function::abstract::Function(f->imageSpace()), f_(f), robot_(robot), ffActive_(ffActive), activeIndex_(activeIndex)
{
  addDirectDependency<JointsSelector>(Output::Value, *f_, Function::Output::Value);
  addDirectDependency<JointsSelector>(Output::Velocity, *f_, Function::Output::Velocity);
  addDirectDependency<JointsSelector>(Output::NormalAcceleration, *f_, Function::Output::NormalAcceleration);

  registerUpdates(Update::Jacobian, &JointsSelector::updateJacobian);
  registerUpdates(Update::JDot, &JointsSelector::updateJDot);

  addOutputDependency<JointsSelector>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<JointsSelector>(Output::JDot, Update::JDot);

  addInputDependency<JointsSelector>(Update::Jacobian, *f_, Function::Output::Jacobian);
  addInputDependency<JointsSelector>(Update::JDot, *f_, Function::Output::JDot);

  if(ffActive_)
  {
    addVariable(robot_->qFreeFlyer(), f_->linearIn(*robot_->qFreeFlyer()));
    jacobian_.at(robot_->qFreeFlyer().get()).setZero();
  }
  if(activeIndex.size())
  {
    addVariable(robot_->qJoints(), f_->linearIn(*robot_->qJoints()));
    jacobian_.at(robot_->qJoints().get()).setZero();
  }
}

void JointsSelector::updateJacobian()
{
  if(ffActive_)
  {
    jacobian_[robot_->qFreeFlyer().get()] = f_->jacobian(*robot_->qFreeFlyer());
  }
  if(activeIndex_.size())
  {
    const auto & jacIn = f_->jacobian(*robot_->qJoints());
    for(const auto & p : activeIndex_)
    {
      jacobian_[robot_->qJoints().get()].middleCols(p.first, p.second) =
          jacIn.block(0, p.first, jacIn.rows(), p.second);
    }
  }
}

void JointsSelector::updateJDot()
{
  if(ffActive_)
  {
    JDot_[robot_->qFreeFlyer().get()] = f_->JDot(*robot_->qFreeFlyer());
  }
  if(activeIndex_.size())
  {
    const auto & JDotIn = f_->JDot(*robot_->qJoints());
    for(const auto & p : activeIndex_)
    {
      JDot_[robot_->qJoints().get()].middleCols(p.first, p.second) = JDotIn.block(0, p.first, JDotIn.rows(), p.second);
    }
  }
}

} // namespace robot

} // namespace tvm
