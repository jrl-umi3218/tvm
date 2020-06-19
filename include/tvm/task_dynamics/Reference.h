/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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

namespace tvm::task_dynamics
{
  /** Compute e^(k)* = ref (at given k-th order). */
  class TVM_DLLAPI Reference : public abstract::TaskDynamics
  {
  public:
    class TVM_DLLAPI Impl : public abstract::TaskDynamicsImpl
    {
    public:
      Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, Order order, FunctionPtr ref);
      void updateValue() override;

      ~Impl() override = default;

      /** Getter on the reference function. */
      FunctionPtr ref() const { return ref_; }
      /** Setter on the reference function. */
      void ref(const FunctionPtr& r);

    private:
      void setReference(const FunctionPtr& ref);

      FunctionPtr ref_;
    };

    /** \param order The order of derivation k of the error that need to follow the reference
      * \param ref The reference function. 
      */
    Reference(Order order, const FunctionPtr& ref);

    ~Reference() override = default;

  protected:
    std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;
    Order order_() const override;

  private:
    Order refOrder_;
    FunctionPtr ref_;
  };

  /** Compute dot{e}* = ref (kinematic order). 
    *
    * This is a simple conveniency shortcut for a Reference instance with order Order::One
    */
  class TVM_DLLAPI ReferenceVelocity : public Reference
  {
  public:
    ReferenceVelocity(const FunctionPtr& ref);
  };

  /** Compute ddot{e}* = ref (dynamic order). 
    *
    * This is a simple conveniency shortcut for a Reference instance with order Order::Two
    */
  class TVM_DLLAPI ReferenceAcceleration : public Reference
  {
  public:
    ReferenceAcceleration(const FunctionPtr& ref);
  };
}
