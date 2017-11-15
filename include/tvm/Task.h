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

#include <memory>

namespace tvm
{
  /** A conveniency proxy to represents expression f==0, f>=0 or f<=0 where f
    * is a function
    */
  class TVM_DLLAPI ProtoTask
  {
  public:
    FunctionPtr f_;
    constraint::Type type_;
  };

  /** Convenient operators to form ProtoTask. For now, we only accept rhs=0
    *
    * Note that you explicitely need to write 0., otherwise the compiler won't
    * be able to decide wich overload to pick between this and shared_ptr
    * operator.
    * (and it is not possible to have an overload with "int rhs", for the same
    * reason)
    */
  ProtoTask TVM_DLLAPI operator==(FunctionPtr f, double rhs);
  ProtoTask TVM_DLLAPI operator>=(FunctionPtr f, double rhs);
  ProtoTask TVM_DLLAPI operator<=(FunctionPtr f, double rhs);


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
    Task(ProtoTask proto, TaskDynamicsPtr td);

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
