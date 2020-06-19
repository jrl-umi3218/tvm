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

// std::variant is basically unusable in clang 6 + libstdc++ due to a bug
// see, e.g. https://bugs.llvm.org/show_bug.cgi?id=33222
// We use https://github.com/mpark/variant/ instead
#include <mpark/variant.hpp>

#include <tvm/defs.h>
#include <tvm/function/abstract/Function.h>
#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm::task_dynamics
{
  /** Given a task dynamics value \f$ e^{(k)*} \f$, compute the new value
    * \f$ e_c^{(k)*} = s e^{(k)*} \f$, \f$ s \in [0, 1] \f$, such that
    * \f$ b_{min} \leq \ e_c^{(k)*} \leq b_{max}\f$ where \f$ b_{min} \f$ and
    * \f$ b_{max} \f$ are given bounds, specified as scalars or vectors.
    */
  template <class TD, class TDImpl = typename TD::Impl>
  class Clamped : public abstract::TaskDynamics
  {
  public:
    using Bounds = mpark::variant<double, Eigen::VectorXd>;

    class Impl : public abstract::TaskDynamicsImpl
    {
    public:
      Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, const TD& innerTaskDynamics, const Bounds& min, const Bounds& max);
      void updateValue() override;
      ~Impl() override = default;

      /** Access to the task dynamics being clamped. */
      const std::shared_ptr<TDImpl>& inner() const;

      /** Access to \f$ b_{min} \f$. */
      const Eigen::VectorXd& min() const { return min_; }
      /** Access to \f$ b_{min} \f$. 
        *
        * \warning It is your responsibility to give a valid \f$ b_{min} \f$ i.e.
        *  * correct size
        *  * \f$ b_{min} \leq 0 \leq b_{max}\f$
        */
      Eigen::VectorXd& min() { return min_; }
      /** Access to \f$ b_{max} \f$. */
      const Eigen::VectorXd& max() const { return max_; }
      /** Access to \f$ b_{max} \f$.
        *
        * \warning It is your responsibility to give a valid \f$ b_{max} \f$ i.e.
        *  * correct size
        *  * \f$ b_{min} \leq 0 \leq b_{max}\f$
        */
      Eigen::VectorXd& max() { return max_; }

    private:
      std::shared_ptr<TDImpl> innerTaskDynamicsImpl_;
      Eigen::VectorXd min_;
      Eigen::VectorXd max_;
    };

    /** Constructor with \f$ b_{min} = -b_{max}\f$  (scalar version).
      *
      * \param innerTaskDynamics The task dynamics to clamp.
      * \param max The maximum value that a component of \f$ e_c^{(k)*} \f$ can
      *        have, in absolute value (\f$ b_{max}\f$).
      */
    Clamped(const TD& innerTaskDynamics, double max);
    /** Constructor with \f$ b_{min} = -b_{max}\f$  (scalar version).
      *
      * \param innerTaskDynamics The task dynamics to clamp.
      * \param max The maximum value that a component of \f$ e_c^{(k)*} \f$ can
      *        have (\f$ b_{max}\f$).
      * \param min The minimum value that a component of \f$ e_c^{(k)*} \f$ can
      *        have (\f$ b_{min}\f$). We need \f$ b_{min} \leq 0 \leq b_{max}\f$.
      */
    Clamped(const TD& innerTaskDynamics, double min, double max);
    /** Constructor with \f$ b_{min} = -b_{max}\f$  (vector version).
      *
      * \param innerTaskDynamics The task dynamics to clamp.
      * \param max The maximum value that a component of \f$ e_c^{(k)*} \f$ can
      *        have, in absolute value (\f$ b_{max}\f$).
      */
    Clamped(const TD& innerTaskDynamics, const VectorConstRef& max);
    /** Constructor with \f$ b_{min} = -b_{max}\f$  (vector version).
      *
      * \param innerTaskDynamics The task dynamics to clamp.
      * \param max The maximum value that a component of \f$ e_c^{(k)*} \f$ can
      *        have (\f$ b_{max}\f$).
      * \param min The minimum value that a component of \f$ e_c^{(k)*} \f$ can
      *        have (\f$ b_{min}\f$). We need \f$ b_{min} \leq 0 \leq b_{max}\f$.
      */
    Clamped(const TD& innerTaskDynamics, const VectorConstRef& min, const VectorConstRef& max);

    ~Clamped() override = default;

  protected:
    std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;
    Order order_() const override;

  private:
    TD innerTaskDynamics_;
    Bounds min_;
    Bounds max_;
  };

  template<class TD, class TDImpl>
  inline Clamped<TD, TDImpl>::Clamped(const TD& innerTaskDynamics, double max)
    : Clamped<TD, TDImpl>(innerTaskDynamics, -max, max)
  {
  }

  template<class TD, class TDImpl>
  inline Clamped<TD, TDImpl>::Clamped(const TD& innerTaskDynamics, double min, double max)
    : innerTaskDynamics_(innerTaskDynamics)
    , min_(min)
    , max_(max)
  {
    if (min > 0)
    {
      throw std::runtime_error("[task_dynamics::Clamped] Minimum values must be negative.");
    }
    if (max < 0)
    {
      throw std::runtime_error("[task_dynamics::Clamped] Maximum values must be positive.");
    }
  }
  
  template<class TD, class TDImpl>
  inline Clamped<TD, TDImpl>::Clamped(const TD& innerTaskDynamics, const VectorConstRef& max)
    : Clamped<TD>(innerTaskDynamics, -max, max)
  {
  }

  template<class TD, class TDImpl>
  inline Clamped<TD, TDImpl>::Clamped(const TD& innerTaskDynamics, const VectorConstRef& min, const VectorConstRef& max)
    : innerTaskDynamics_(innerTaskDynamics)
    , min_(min)
    , max_(max)
  {
    if (min.size() !=  max.size())
    {
      throw std::runtime_error("[task_dynamics::Clamped] The minimum and maximum must have the same size.");
    }
    if ((min.array() > 0).any())
    {
      throw std::runtime_error("[task_dynamics::Clamped] Minimum values must be negative.");
    }
    if ((max.array() < 0).any())
    {
      throw std::runtime_error("[task_dynamics::Clamped] Maximum values must be positive.");
    }
  }
  
  template<class TD, class TDImpl>
  inline std::unique_ptr<abstract::TaskDynamicsImpl> Clamped<TD, TDImpl>::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
  {
    return std::make_unique<Impl>(f, t, rhs, innerTaskDynamics_, min_, max_);
  }

  template<class TD, class TDImpl>
  inline Order Clamped<TD, TDImpl>::order_() const
  {
    return innerTaskDynamics_.order();
  }

  template<class TD, class TDImpl>
  inline Clamped<TD, TDImpl>::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, const TD& innerTaskDynamics, const Bounds& min, const Bounds& max)
    : TaskDynamicsImpl(innerTaskDynamics.order(), f, t, rhs)
    , innerTaskDynamicsImpl_(static_cast<TDImpl*>(innerTaskDynamics.impl(f, t, rhs).release()))
    , min_(min.index() == 1 ? mpark::get<Eigen::VectorXd>(min) : Eigen::VectorXd::Constant(f->size(), mpark::get<double>(min)))
    , max_(max.index() == 1 ? mpark::get<Eigen::VectorXd>(max) : Eigen::VectorXd::Constant(f->size(), mpark::get<double>(max)))
  {
    if (min_.size() != f->size() || max_.size() != f->size())
    {
      throw std::runtime_error("[task_dynamics::Clamped::Impl] Sizes of the minimum, maximum and function must be the same.");
    }
    if ((min_.array() > 0).any())
    {
      throw std::runtime_error("[task_dynamics::Clamped::Impl] Minimum values must be negative.");
    }
    if ((max_.array() < 0).any())
    {
      throw std::runtime_error("[task_dynamics::Clamped::Impl] Maximum values must be positive.");
    }
    addInput(innerTaskDynamicsImpl_, TaskDynamicsImpl::Output::Value);
    addInputDependency(Update::UpdateValue, innerTaskDynamicsImpl_, TaskDynamicsImpl::Output::Value);
  }


  template<class TD, class TDImpl>
  inline void Clamped<TD, TDImpl>::Impl::updateValue()
  {
    const auto& innerValue = innerTaskDynamicsImpl_->value();

    double s = 1;
    for (int i = 0; i < function().size(); ++i)
    {
      if (innerValue[i] > max_[i])
      {
        // innerValue[i] > max_[i] >= 0 so that innerValue[i] != 0
        s = std::min(s, max_[i] / innerValue[i]);
      }
      else if (innerValue[i] < min_[i])
      {
        // innerValue[i] < min_[i] <= 0 so that innerValue[i] != 0
        s = std::min(s, min_[i] / innerValue[i]);
      }
    }

    value_ = s * innerValue;
  }
  
  template<class TD, class TDImpl>
  inline const std::shared_ptr<TDImpl>& Clamped<TD, TDImpl>::Impl::inner() const
  {
    return innerTaskDynamicsImpl_;
  }
}
