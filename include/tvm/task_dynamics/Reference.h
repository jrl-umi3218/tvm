/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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

    TASK_DYNAMICS_DERIVED_FACTORY(refOrder_, ref_)
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
