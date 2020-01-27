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

#include <tvm/task_dynamics/ProportionalDerivative.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{

  ProportionalDerivative::ProportionalDerivative(double kp, double kv)
    : kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(double kp)
    : ProportionalDerivative(kp, 2 * std::sqrt(kp))
  {
  }

  std::unique_ptr<abstract::TaskDynamicsImpl> ProportionalDerivative::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
  {
    return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, t, rhs, kp_, kv_));
  }

  ProportionalDerivative::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, double kp, double kv)
    : TaskDynamicsImpl(Order::Two, f, t, rhs)
    , kp_(kp)
    , kv_(kv)
  {
  }

  void ProportionalDerivative::Impl::updateValue()
  {
    value_ = -kv_ * function().velocity() - kp_ * (function().value() - rhs());
  }

  std::pair<double, double> ProportionalDerivative::Impl::gains() const
  {
    return {kp_, kv_};
  }

  void ProportionalDerivative::Impl::gains(double kp, double kv)
  {
    kp_ = kp;
    kv_ = kv;
  }

  void ProportionalDerivative::Impl::gains(double kp)
  {
    kp_ = kp;
    kv_ = 2 * std::sqrt(kp);
  }

}  // namespace task_dynamics

}  // namespace tvm
