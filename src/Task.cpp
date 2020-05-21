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

#include <tvm/Task.h>

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

#include <stdexcept>


namespace tvm
{
  Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td)
    : f_(f)
    , type_(t)
    , td_(td.impl(f, t, Eigen::VectorXd::Zero(f->size())))
  {
    if (t == constraint::Type::DOUBLE_SIDED)
      throw std::runtime_error("Double sided tasks need to have non-zero bounds.");
  }

  Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td, double rhs)
    : Task(f, t, td, Eigen::VectorXd::Constant(f->size(),rhs))
  {
  }

  Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td, const Eigen::VectorXd & rhs)
    : f_(f)
    , type_(t)
    , td_(td.impl(f, t, rhs))
  {
    if (t == constraint::Type::DOUBLE_SIDED)
      throw std::runtime_error("Double sided tasks need to have two bounds.");
  }

  Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td, double l, double u)
    : Task(f, t, td, Eigen::VectorXd::Constant(f->size(), l), Eigen::VectorXd::Constant(f->size(), u))
  {
  }

  Task::Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td, const Eigen::VectorXd & l, const Eigen::VectorXd & u)
    : f_(f)
    , type_(t)
    , td_(td.impl(f, constraint::Type::GREATER_THAN, l))
    , td2_(td.impl(f, constraint::Type::LOWER_THAN, u))
  {
    if (t != constraint::Type::DOUBLE_SIDED)
      throw std::runtime_error("This constructor is for double sided constraints only.");
  }

  Task::Task(utils::ProtoTaskEQ proto, const task_dynamics::abstract::TaskDynamics& td)
    : Task(proto.f_, constraint::Type::EQUAL, td, proto.rhs_.toVector(proto.f_->size()))
  {
  }

  Task::Task(utils::ProtoTaskLT proto, const task_dynamics::abstract::TaskDynamics& td)
    : Task(proto.f_, constraint::Type::LOWER_THAN, td, proto.rhs_.toVector(proto.f_->size()))
  {
  }

  Task::Task(utils::ProtoTaskGT proto, const task_dynamics::abstract::TaskDynamics& td)
    : Task(proto.f_, constraint::Type::GREATER_THAN, td, proto.rhs_.toVector(proto.f_->size()))
  {
  }

  Task::Task(utils::ProtoTaskDS proto, const task_dynamics::abstract::TaskDynamics& td)
    : Task(proto.f_, constraint::Type::DOUBLE_SIDED, td, 
            proto.l_.toVector(proto.f_->size()), proto.u_.toVector(proto.f_->size()))
  {
  }


  FunctionPtr Task::function() const
  {
    return f_;
  }

  constraint::Type Task::type() const
  {
    return type_;
  }

  TaskDynamicsPtr Task::taskDynamics() const
  {
    return td_;
  }

  TaskDynamicsPtr Task::secondBoundTaskDynamics() const
  {
      return td2_;
  }

}  // namespace tvm
