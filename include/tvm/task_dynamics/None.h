/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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

      TASK_DYNAMICS_DERIVED_FACTORY()
    };

  }  // namespace task_dynamics

}  // namespace tvm
