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

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm
{

  namespace task_dynamics
  {

    /** Compute e* = -f(0) + rhs (Geometric order). For linear functions only. */
    class TVM_DLLAPI None : public abstract::TaskDynamics
    {
    public:
      class TVM_DLLAPI Impl: public abstract::TaskDynamicsImpl
      {
      public:
        Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs);
        void updateValue() override;
        ~Impl() override = default;

      private:
        const function::abstract::LinearFunction* lf_;
      };

      ~None() override = default;

    protected:
      std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;
      Order order_() const override;
    };

  }  // namespace task_dynamics

}  // namespace tvm
