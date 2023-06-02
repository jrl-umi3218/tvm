/*
 * Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#pragma once

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm
{

namespace task_dynamics
{

/** Compute \f$ e^{(d)*} = - \sum_{i=0}^{d-1} \frac{d!}{i! dt^{k-i}} \f$ (Order d)
 *  with \f$ dt \f$ an integration step. That is, compute the desired d-th derivative of \f$ e \f$
 *  such that if\f$ e(t+dt) \f$ was equal to its Taylor expansion of order d, achieving
 *  \f$ e^{(d)*} \f$ would bring \f$ e \f$ to 0 in on step \f$ dt \f$.
 */
class TVM_DLLAPI OneStepToZero : public abstract::TaskDynamics
{
public:
  static void checkParam(Order d, double dt);

  class TVM_DLLAPI Impl : public abstract::TaskDynamicsImpl
  {
  public:
    Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, Order d, double dt);
    void updateValue() override;

    ~Impl() override = default;

  private:
    double dt_;
  };

  OneStepToZero(Order d, double dt);

protected:
  std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f,
                                                    constraint::Type t,
                                                    const Eigen::VectorXd & rhs) const override;
  Order order_() const override;

  TASK_DYNAMICS_DERIVED_FACTORY(d_, dt_)

private:
  Order d_;
  double dt_;
};

} // namespace task_dynamics

} // namespace tvm
