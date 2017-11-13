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

#include <Eigen/Core>

#include <memory>

//FIXME add mechanisms for when the function's output is resized
//FIXME Consider the possibility of having variables in task dynamics?

namespace tvm
{

namespace task_dynamics
{

namespace abstract
{

  class TVM_DLLAPI TaskDynamics : public graph::abstract::Node<TaskDynamics>
  {
  public:
    SET_OUTPUTS(TaskDynamics, Value)
    SET_UPDATES(TaskDynamics, UpdateValue)

    void setFunction(FunctionPtr f);

    const Eigen::VectorXd& value() const;
    Order order() const;

    virtual void updateValue() = 0;

  protected:
    TaskDynamics(Order order);
    const function::abstract::Function & function() const;

    /** Hook for derived class, called at the end of setFunction.*/
    virtual void setFunction_();

    Eigen::VectorXd value_;

  private:
    Order order_;
    FunctionPtr f_;
  };

  inline const Eigen::VectorXd& TaskDynamics::value() const
  {
    return value_;
  }

  inline Order TaskDynamics::order() const
  {
    return order_;
  }

  inline const function::abstract::Function & TaskDynamics::function() const
  {
    assert(f_);
    return *f_;
  }

  inline void TaskDynamics::setFunction_()
  {
  }

}  // namespace abstract

}  // namespace task_dynamics

}  // namespace tvm
