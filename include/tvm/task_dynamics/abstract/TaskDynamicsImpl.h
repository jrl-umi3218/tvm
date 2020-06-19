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
