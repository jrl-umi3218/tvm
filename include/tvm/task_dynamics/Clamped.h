/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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
  class Clamped : public TD
  {
  public:
    using Bounds = mpark::variant<double, Eigen::VectorXd>;

    class Impl : public TDImpl
    {
    public:
      template<typename ... Args>
      Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, const Bounds& min, const Bounds& max, Args&& ... args);
      void updateValue() override;
      ~Impl() override = default;

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
      Eigen::VectorXd min_;
      Eigen::VectorXd max_;
    };

    /** Constructor with \f$ b_{min} = -b_{max}\f$  (scalar version).
      *
      * \param max The maximum value that a component of \f$ e_c^{(k)*} \f$ can
      *        have, in absolute value (\f$ b_{max}\f$).
      * \param args These are forwarded to the TD constructor
      */
    template<typename ... Args>
    Clamped(double max, Args&& ... args);

    /** Constructor with \f$b_{min}\f$ and \f$b_{max}\f$  (scalar version).
      *
      * \param minMax The minimum (\f$b_{min}\f$) and maximum (\f$b_{max}\f$)
      *               value that a component of \f$ e_c^{(k)*}\f$ can have. We require that
      *               \f$b_{min} \leq 0 \leq b_{max}\f$.
      * \param args These are forwarded to the TD constructor
      */
    template<typename ... Args>
    Clamped(const std::pair<double, double>& minMax, Args&& ... args);

    /** Constructor with \f$ b_{min} = -b_{max}\f$  (vector version).
      *
      * \param max The maximum value that a component of \f$ e_c^{(k)*} \f$ can
      *        have, in absolute value (\f$ b_{max}\f$).
      * \param args These are forwarded to the TD constructor
      */
    template<typename ... Args>
    Clamped(const VectorConstRef& max, Args&& ... args);

    /** Constructor with \f$b_{min}\f$ and \f$b_{max}\f$  (vector version).
      *
      * \param minMax The minimum (\f$b_{min}\f$) and maximum (\f$b_{max}\f$)
      *               value that a component of \f$ e_c^{(k)*}\f$ can have. We require that
      *               \f$b_{min} \leq 0 \leq b_{max}\f$.
      * \param args These are forwarded to the TD constructor
      */
    template<typename ... Args>
    Clamped(const std::pair<VectorConstRef, VectorConstRef>& minMax, Args&& ... args);

    ~Clamped() override = default;

  protected:
    std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override
    {
      return TD::template impl_<Impl>(f, t, rhs, min_, max_);
    }

    COMPOSABLE_TASK_DYNAMICS_DERIVED_FACTORY(TD, min_, max_)

  private:
    Bounds min_;
    Bounds max_;
  };

  template<class TD, class TDImpl>
  template<typename ... Args>
  inline Clamped<TD, TDImpl>::Clamped(double max, Args&& ... args)
    : Clamped<TD, TDImpl>({-max, max}, std::forward<Args>(args)...)
  {
  }

  template<class TD, class TDImpl>
  template<typename ... Args>
  inline Clamped<TD, TDImpl>::Clamped(const std::pair<double, double> & minMax, Args&& ... args)
    : TD(std::forward<Args>(args)...),
      min_(minMax.first)
    , max_(minMax.second)
  {
    const auto & min = minMax.first;
    const auto & max = minMax.second;
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
  template<typename ... Args>
  inline Clamped<TD, TDImpl>::Clamped(const VectorConstRef& max, Args&& ... args)
    : Clamped<TD>({-max, max}, std::forward<Args>(args)...)
  {
  }

  template<class TD, class TDImpl>
  template<typename ... Args>
  inline Clamped<TD, TDImpl>::Clamped(const std::pair<VectorConstRef, VectorConstRef>& minMax, Args&& ... args)
    : TD(std::forward<Args>(args)...)
    , min_(minMax.first)
    , max_(minMax.second)
  {
    const auto & min = minMax.first;
    const auto & max = minMax.second;
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
  template<typename ... Args>
  inline Clamped<TD, TDImpl>::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, const Bounds& min, const Bounds& max, Args&& ... args)
    : TDImpl(f, t, rhs, std::forward<Args>(args)...)
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
  }


  template<class TD, class TDImpl>
  inline void Clamped<TD, TDImpl>::Impl::updateValue()
  {
    TDImpl::updateValue();
    double s = 1;
    for (int i = 0; i < this->function().size(); ++i)
    {
      if (this->value_[i] > max_[i])
      {
        // innerValue[i] > max_[i] >= 0 so that innerValue[i] != 0
        s = std::min(s, max_[i] / this->value_[i]);
      }
      else if (this->value_[i] < min_[i])
      {
        // this->value_[i] < min_[i] <= 0 so that this->value_[i] != 0
        s = std::min(s, min_[i] / this->value_[i]);
      }
    }
    this->value_ *= s;
  }

} // namespace task_dynamics
