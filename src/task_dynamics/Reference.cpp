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

#include <tvm/task_dynamics/Reference.h>

#include <tvm/function/abstract/Function.h>
#include <tvm/task_dynamics/Reference.h>

namespace tvm::task_dynamics
{
  Reference::Reference(Order order, const FunctionPtr& ref)
    : refOrder_(order)
    , ref_(ref)
  {
  }

  std::unique_ptr<abstract::TaskDynamicsImpl> Reference::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
  {
    return std::make_unique<Impl>(f, t, rhs, refOrder_, ref_);
  }

  Order Reference::order_() const
  {
    return refOrder_;
  }


  Reference::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, Order order, FunctionPtr ref)
    : TaskDynamicsImpl(order, f, t, rhs)
  {
    setReference(ref);
  }

  void Reference::Impl::updateValue()
  {
    value_ = ref_->value();
  }

  void Reference::Impl::ref(const FunctionPtr& r)
  {
    // Changing the reference requires recomputing the computation graph
    // We do not have the proper "event" system to handle that for now
    throw std::runtime_error("[task_dynamics::Reference::Impl::ref] Unimplemented function");
  }

  void Reference::Impl::setReference(const FunctionPtr& ref)
  {
    if (!ref)
    {
      throw std::runtime_error("[task_dynamics::Reference::Impl] You cannot pass a nullptr as reference function.");
    }

    ref_ = ref;
    addInput(ref, internal::FirstOrderProvider::Output::Value);
    addInputDependency(Update::UpdateValue, ref, internal::FirstOrderProvider::Output::Value);
  }

  ReferenceVelocity::ReferenceVelocity(const FunctionPtr& ref)
    : Reference(Order::One, ref)
  {
  }

  ReferenceAcceleration::ReferenceAcceleration(const FunctionPtr& ref)
    : Reference(Order::Two, ref)
  {
  }
}