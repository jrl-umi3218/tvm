/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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
