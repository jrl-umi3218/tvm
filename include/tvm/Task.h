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

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/constraint/enums.h>
#include <tvm/constraint/internal/RHSVectors.h>
#include <tvm/utils/ProtoTask.h>
#include <tvm/task_dynamics/abstract/TaskDynamics.h>

#include <memory>

namespace tvm
{
  /** A task is a triplet (Function, operator, TaskDynamics) where operator is
    * ==, >= or <=*/
  class TVM_DLLAPI Task
  {
  public:
    Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td);
    Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td, double rhs);
    Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td,
         const Eigen::VectorXd& rhs);
    Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td, double l, double u);
    Task(FunctionPtr f, constraint::Type t, const task_dynamics::abstract::TaskDynamics& td,
         const Eigen::VectorXd& l, const Eigen::VectorXd& u);
    Task(utils::ProtoTaskEQ proto, const task_dynamics::abstract::TaskDynamics& td);
    Task(utils::ProtoTaskLT proto, const task_dynamics::abstract::TaskDynamics& td);
    Task(utils::ProtoTaskGT proto, const task_dynamics::abstract::TaskDynamics& td);
    Task(utils::ProtoTaskDS proto, const task_dynamics::abstract::TaskDynamics& td);

    FunctionPtr function() const;
    constraint::Type type() const;
    TaskDynamicsPtr taskDynamics() const;
    TaskDynamicsPtr secondBoundTaskDynamics() const;  //the dynamics of the upper bound, in the case of double-sided task only.

    template<typename T, typename TDImpl = typename T::Impl>
    std::shared_ptr<TDImpl> taskDynamics() const;

    template<typename T>
    std::shared_ptr<typename T::Impl> secondBoundTaskDynamics() const;

  private:
    FunctionPtr f_;
    constraint::Type type_;
    TaskDynamicsPtr td_;
    TaskDynamicsPtr td2_ = nullptr;             //used only for double sided tasks, as dynamics for upper bound.
  };


  template<typename T, typename TDImpl>
  std::shared_ptr<TDImpl> Task::taskDynamics() const
  {
    if (td_->checkType<TDImpl>())
      return std::static_pointer_cast<TDImpl>(td_);
    else
      throw std::runtime_error("Unable to cast the task dynamics into the desired type.");
  }

  template<typename T>
  std::shared_ptr<typename T::Impl> Task::secondBoundTaskDynamics() const
  {
    if (td2_->checkType<T>())
      return std::static_pointer_cast<typename T::Impl>(td2_);
    else
      throw std::runtime_error("Unable to cast the task dynamics into the desired type.");
  }

}  // namespace tvm
