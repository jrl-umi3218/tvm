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

// std::variant is basically unusable in clang 6 + libstdc++ 
// see, e.g. https://bugs.llvm.org/show_bug.cgi?id=33222
// We use a workaround for std::variant<double, Eigen::VectorXd, Eigen::MatrixXd>
// in this case.
#if defined(__clang__) && __clang_major__ < 7 && defined(__GLIBCXX__)
#include <tvm/task_dynamics/internal/workarounds.h>
#else
#include <variant>
#endif

#include <tvm/defs.h>
#include <tvm/function/abstract/Function.h>
#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm::task_dynamics
{
  template <class TD>
  class Clamped : public abstract::TaskDynamics
  {
  public:
#if defined(__clang__) && __clang_major__ < 7 && defined(__GLIBCXX__)    
    using Bounds = internal::WorkaroundDoubleVectorMatrixVariant; 
#else
    using Bounds = std::variant<double, Eigen::VectorXd>;
#endif

    class Impl : public abstract::TaskDynamicsImpl
    {
    public:
      Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, const TD& innerTaskDynamics, const Bounds& min, const Bounds& max);
      void updateValue() override;

      std::shared_ptr<typename TD::Impl> inner() const;

    private:
      std::shared_ptr<typename TD::Impl> innerTaskDynamicsImpl_;
      Eigen::VectorXd min_;
      Eigen::VectorXd max_;
    };

    Clamped(const TD& innerTaskDynamics, double max);
    Clamped(const TD& innerTaskDynamics, double min, double max);
    Clamped(const TD& innerTaskDynamics, const VectorConstRef& max);
    Clamped(const TD& innerTaskDynamics, const VectorConstRef& min, const VectorConstRef& max);

  protected:
    std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;
    Order order_() const override;

  private:
    TD innerTaskDynamics_;
    Bounds min_;
    Bounds max_;
  };

  template<class TD>
  inline Clamped<TD>::Clamped(const TD& innerTaskDynamics, double max)
    : Clamped<TD>(innerTaskDynamics, -max, max)
  {
  }

  template<class TD>
  inline Clamped<TD>::Clamped(const TD& innerTaskDynamics, double min, double max)
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
  
  template<class TD>
  inline Clamped<TD>::Clamped(const TD& innerTaskDynamics, const VectorConstRef& max)
    : Clamped<TD>(innerTaskDynamics, -max, max)
  {
  }

  template<class TD>
  inline Clamped<TD>::Clamped(const TD& innerTaskDynamics, const VectorConstRef& min, const VectorConstRef& max)
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
  
  template<class TD>
  inline std::unique_ptr<abstract::TaskDynamicsImpl> Clamped<TD>::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
  {
    return std::unique_ptr<abstract::TaskDynamicsImpl>(new Impl(f, t, rhs, innerTaskDynamics_, min_, max_));
  }

  template<class TD>
  inline Order Clamped<TD>::order_() const
  {
    return innerTaskDynamics_.order();
  }

  template<class TD>
  inline Clamped<TD>::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, const TD& innerTaskDynamics, const Bounds& min, const Bounds& max)
    : TaskDynamicsImpl(innerTaskDynamics.order(), f, t, rhs)
    , innerTaskDynamicsImpl_(static_cast<typename TD::Impl*>(innerTaskDynamics.impl(f, t, rhs).release()))
    , min_(min.index() == 1 ? std::get<Eigen::VectorXd>(min) : VectorXd::Constant(f->size(), std::get<double>(min)))
    , max_(max.index() == 1 ? std::get<Eigen::VectorXd>(max) : VectorXd::Constant(f->size(), std::get<double>(max)))
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


  template<class TD>
  inline void Clamped<TD>::Impl::updateValue()
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
  
  template<class TD>
  inline std::shared_ptr<typename TD::Impl> Clamped<TD>::Impl::inner() const
  {
    return innerTaskDynamicsImpl_;
  }
}