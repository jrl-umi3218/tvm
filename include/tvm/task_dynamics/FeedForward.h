/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#pragma once

#include <tvm/task_dynamics/Proportional.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>

namespace tvm
{

namespace task_dynamics
{

  /** Given a task dynamics value \f$ e^{(k)*} \f$, compute the new value \f$
   * e_ff^{(k)*} = e^{(k)*} - ff \f$, \f$ ff \f$ is data provided by another
   * TVM node of the same size
    */
  template <class TD, class TDImpl = typename TD::Impl>
  class FeedForward : public TD
  {
  public:

    using getFeedForwardT = std::function<const Eigen::VectorXd&()>;
    using makeImplT = std::function<std::unique_ptr<abstract::TaskDynamicsImpl>(const FeedForward&, FunctionPtr, constraint::Type, const Eigen::VectorXd&)>;

    template<class FFProvider, class FFSignal, typename ... Args>
    FeedForward(std::shared_ptr<FFProvider> provider, const Eigen::VectorXd & (FFProvider::*method)() const,
                FFSignal signal, Args && ... args)
    : TD(std::forward<Args>(args)...)
    {
      feedForward_ = [provider, method]() -> const Eigen::VectorXd & {
        return ((provider.get())->*method)();
      };
      makeImpl_ = [provider, signal](const FeedForward & self, FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) {
        auto impl = std::make_unique<Impl>(static_cast<TDImpl&&>(*self.TD::impl_(f, t, rhs).release()), self.feedForward_);
        impl->addInputDependency(TDImpl::Update::UpdateValue, provider, signal);
        return impl;
      };
    }

    ~FeedForward() override = default;

    class Impl : public TDImpl
    {
    public:
      friend class FeedForward;

      Impl(TDImpl&& base, const getFeedForwardT& ff) : TDImpl(base), feedForward_(ff)
      {
        if(feedForward_().size() != this->function().size())
        {
          throw std::runtime_error("[task_dynamics::FeedForward] Feed forward term does not have the same size as the provided function");
        }
      }

      ~Impl() override = default;

      void updateValue() override
      {
        TDImpl::updateValue();
        this->value_ -= feedForward_();
      }
    private:
      getFeedForwardT feedForward_;
    };

  protected:
    getFeedForwardT feedForward_;
    makeImplT makeImpl_;
    std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override
    {
      return makeImpl_(*this, f, t, rhs);
    }
  };

  using FeedForwardPD = FeedForward<ProportionalDerivative>;
  using FeedForwardP = FeedForward<Proportional>;


} // namespace task_dynamics

} // namespace tvm
