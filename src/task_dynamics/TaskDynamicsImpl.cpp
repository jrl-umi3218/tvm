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

#include <tvm/task_dynamics/abstract/TaskDynamicsImpl.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

  namespace task_dynamics
  {

    namespace abstract
    {

      TaskDynamicsImpl::TaskDynamicsImpl(Order order, FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs)
        : order_(order)
        , type_(t)
        , rhs_(rhs)
      {
        if (f->size() != rhs.size())
        {
          throw std::runtime_error("[TaskDynamicsImpl] rhs does not have the same size as f.");
        }
        setFunction(f);
        registerUpdates(Update::UpdateValue, &TaskDynamicsImpl::updateValue);
        addOutputDependency(Output::Value, Update::UpdateValue);
      }

      void TaskDynamicsImpl::setFunction(FunctionPtr f)
      {
        if (f)
        {
          f_ = f;
          addInput(f, internal::FirstOrderProvider::Output::Value); //FIXME it's not great to have to resort to internal::FirstOrderProvider
          addInputDependency(Update::UpdateValue, f, internal::FirstOrderProvider::Output::Value);
          if (order_ == Order::Two)
          {
            addInput(f, function::abstract::Function::Output::Velocity);
            addInputDependency(Update::UpdateValue, f, function::abstract::Function::Output::Velocity);
          }
          value_.resize(f->size());
        }
        else
          throw std::runtime_error("You cannot pass a nullptr as a function.");
      }

    }  // namespace abstract

  }  // namespace task_dynamics

}  // namespace tvm
