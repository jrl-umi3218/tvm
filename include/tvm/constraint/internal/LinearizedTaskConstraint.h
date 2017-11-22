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

#include <tvm/Task.h>
#include <tvm/constraint/abstract/LinearConstraint.h>
#include <tvm/utils/ProtoTask.h>

namespace tvm
{

namespace constraint
{

namespace internal
{

  /** Given a task (e, op, rhs, e*), this class derives the constraint
    * d^k e/dt^k op  e*(e,de/dt,...de^{k-1}/dt^{k-1}, rhs [,g]), where e is an
    * error function, op is ==, >= or <= and e* is a desired error dynamics. 
    * k is specified by e* and (optional) g is any other quantities.
    *
    * FIXME Consider the case where the TaskDynamics has its own variables?
    *
    * EQUAL case
    * \dot
    * digraph "update graph" {
    *   rankdir="LR";
    *   {
    *     rank=same; node [shape=circle];
    *     f; td;
    *   }
    *   {
    *     uValue [label=Value];
    *     updateRHS;
    *   }
    *   {
    *     rank = same; node [shape=hexagon];
    *     Value; Jacobian; E;
    *   }
    *   {
    *     rank = same; node [style=invis, label=""];
    *     outValue; outJacobian; outE;
    *   }
    *   Value -> outValue [label="value()"];
    *   Jacobian -> outJacobian [label="jacobian(x_i)"];
    *   E -> outE [label="e()"];
    *   x_i -> uValue [label="value()"];
    *   uValue -> Value;
    * }
    * \enddot
    */
  class TVM_DLLAPI LinearizedTaskConstraint : public abstract::LinearConstraint
  {
  public:
    SET_UPDATES(LinearizedTaskConstraint, UpdateRHS, UpdateRHS2)

    LinearizedTaskConstraint(const Task& task);
    template<constraint::Type T>
    LinearizedTaskConstraint(const utils::ProtoTask<T>& pt, TaskDynamicsPtr td);

    void updateLKin();
    void updateLDyn();
    void updateUKin();
    void updateUDyn();
    void updateEKin();
    void updateEDyn();
    void updateU2Kin();
    void updateU2Dyn();

    const tvm::internal::MatrixWithProperties& jacobian(const Variable& x) const override;

  private:
    FunctionPtr f_;
    TaskDynamicsPtr td_;
    TaskDynamicsPtr td2_; // for double sided constraints only;
  };


  template<constraint::Type T>
  LinearizedTaskConstraint::LinearizedTaskConstraint(const utils::ProtoTask<T>& pt, TaskDynamicsPtr td)
    : LinearizedTaskConstraint(Task(pt, td))
  {
  }

}  // namespace internal

}  // namespace constraint

}  // namespace tvm
