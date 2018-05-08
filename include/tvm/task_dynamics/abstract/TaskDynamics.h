#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

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
