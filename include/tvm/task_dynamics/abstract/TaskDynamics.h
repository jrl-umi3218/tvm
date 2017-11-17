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
#include <typeinfo>

//FIXME add mechanisms for when the function's output is resized
//FIXME Consider the possibility of having variables in task dynamics?

namespace tvm
{

namespace task_dynamics
{

namespace abstract
{
  //forward declaration
  class TaskDynamicsImpl;

  class TVM_DLLAPI TaskDynamics
  {
  public:
    virtual std::unique_ptr<TaskDynamicsImpl> impl(FunctionPtr f) const;

  protected:
    virtual std::unique_ptr<TaskDynamicsImpl> impl_(FunctionPtr f) const = 0;
  };

  class TVM_DLLAPI TaskDynamicsImpl : public graph::abstract::Node<TaskDynamicsImpl>
  {
  public:
    SET_OUTPUTS(TaskDynamicsImpl, Value)
    SET_UPDATES(TaskDynamicsImpl, UpdateValue)

    const Eigen::VectorXd& value() const;
    Order order() const;

    virtual void updateValue() = 0;

    /** Check if this is an instance of T::Impl */
    template<typename T>
    bool checkType() const;

  protected:
    TaskDynamicsImpl(Order order, FunctionPtr f);
    const function::abstract::Function & function() const;

    Eigen::VectorXd value_;

  private:
    void setFunction(FunctionPtr f);

    Order order_;
    FunctionPtr f_;

    //for casting back to the derived type
    size_t typeInfo_;

    friend TaskDynamics;
  };

  inline const Eigen::VectorXd& TaskDynamicsImpl::value() const
  {
    return value_;
  }

  inline Order TaskDynamicsImpl::order() const
  {
    return order_;
  }

  inline const function::abstract::Function & TaskDynamicsImpl::function() const
  {
    assert(f_);
    return *f_;
  }

  template<typename T>
  inline bool TaskDynamicsImpl::checkType() const
  {
    return typeid(T).hash_code() == typeInfo_;
  }

}  // namespace abstract

}  // namespace task_dynamics

}  // namespace tvm
