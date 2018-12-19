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

#include <tvm/defs.h>
#include <tvm/graph/abstract/Node.h>
#include <tvm/task_dynamics/enums.h>
#include <tvm/task_dynamics/abstract/TaskDynamicsImpl.h>

#include <Eigen/Core>

//FIXME add mechanisms for when the function's output is resized
//FIXME Consider the possibility of having variables in task dynamics?

namespace tvm
{

namespace task_dynamics
{

namespace abstract
{
  /** This is a base class to describe how a task is to be regulated, i.e. how
    * to compute e^(d)* for a task with constraint part f op rhs, where f is a
    * function, op is one operator among (==, <=, >=), rhs is a constant or a
    * vector and e = f-rhs. d is the order of the task dynamics.
    *
    * TaskDynamics is a lightweight descriptor, independent of a particular
    * task, that is meant for the end user.
    * Internally, it is turned into a TaskDynamicsImpl when linked to a given
    * function and rhs.
    */
  class TVM_DLLAPI TaskDynamics
  {
  public:
    virtual ~TaskDynamics() = default;

    std::unique_ptr<TaskDynamicsImpl> impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const;

  protected:
    virtual std::unique_ptr<TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const = 0;
  };

}  // namespace abstract

}  // namespace task_dynamics

}  // namespace tvm
