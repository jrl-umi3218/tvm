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

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/constraint/enums.h>
#include <tvm/constraint/internal/RHSVectors.h>
#include <tvm/utils/ProtoTask.h>

#include <memory>

namespace tvm
{
  /** A task is a triplet (Function, operator, TaskDynamics) where operator is
    * ==, >= or <=*/
  class TVM_DLLAPI Task
  {
  public:
    Task(FunctionPtr f, constraint::Type t, TaskDynamicsPtr td);
    Task(FunctionPtr f, constraint::Type t, TaskDynamicsPtr td, double rhs);
    Task(FunctionPtr f, constraint::Type t, TaskDynamicsPtr td, 
         const Eigen::VectorXd& rhs);
    Task(FunctionPtr f, constraint::Type t, TaskDynamicsPtr td, double l, double u);
    Task(FunctionPtr f, constraint::Type t, TaskDynamicsPtr td, 
         const Eigen::VectorXd& l, const Eigen::VectorXd& u);
    Task(utils::ProtoTaskEQ proto, TaskDynamicsPtr td);
    Task(utils::ProtoTaskLT proto, TaskDynamicsPtr td);
    Task(utils::ProtoTaskGT proto, TaskDynamicsPtr td);
    Task(utils::ProtoTaskDS proto, TaskDynamicsPtr td);

    FunctionPtr function() const;
    constraint::Type type() const;
    TaskDynamicsPtr taskDynamics() const;

  private:
    FunctionPtr f_;
    constraint::Type type_;
    TaskDynamicsPtr td_;
    constraint::internal::RHSVectors vectors_;
  };
}  // namespace tvm
