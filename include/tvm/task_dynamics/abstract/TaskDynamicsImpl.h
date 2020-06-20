/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#pragma once

#include <tvm/defs.h>
#include <tvm/constraint/enums.h>
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
      class TaskDynamics;

      /** Base class for the implementation of a task dynamics*/
      class TVM_DLLAPI TaskDynamicsImpl : public graph::abstract::Node<TaskDynamicsImpl>
      {
      public:
        SET_OUTPUTS(TaskDynamicsImpl, Value)
        SET_UPDATES(TaskDynamicsImpl, UpdateValue)

        virtual ~TaskDynamicsImpl() = default;

        const Eigen::VectorXd& value() const;
        Order order() const;

        virtual void updateValue() = 0;

        /** Check if this is an instance of T::Impl */
        template<typename T>
        bool checkType() const;

      protected:
        TaskDynamicsImpl(Order order, FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs);

        /** Access to the task function.*/
        const function::abstract::Function & function() const;
        /** Access to the task operator. */
        constraint::Type type() const;
        /** Access to the task right hand side.*/
        const Eigen::VectorXd& rhs() const;

        /** Cache to store the results of updateValue()*/
        Eigen::VectorXd value_;

      private:
        void setFunction(FunctionPtr f);

        Order order_;
        FunctionPtr f_;
        constraint::Type type_;
        Eigen::VectorXd rhs_;

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

      inline constraint::Type TaskDynamicsImpl::type() const
      {
        return type_;
      }

      inline const Eigen::VectorXd & TaskDynamicsImpl::rhs() const
      {
        return rhs_;
      }

      template<typename T>
      inline bool TaskDynamicsImpl::checkType() const
      {
        return dynamic_cast<const T *>(this) != nullptr;
      }

    }  // namespace abstract

  }  // namespace task_dynamics

}  // namespace tvm
