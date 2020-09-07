// Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm::example
{
/** A class implementing the task dynamics
 * \f$
 *     \dot{e}^* = - (a \exp(-b \left\|e\right\|) + c) e
 * \f$
 */
class AdaptiveProportional : public task_dynamics::abstract::TaskDynamics
{
public:
  class Impl : public task_dynamics::abstract::TaskDynamicsImpl
  {
  public:
    Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs, double a, double b, double c);
    void updateValue() override;

  private:
    double a_;
    double b_;
    double c_;
  };

  AdaptiveProportional(double a, double b, double c);

protected:
  std::unique_ptr<task_dynamics::abstract::TaskDynamicsImpl> impl_(FunctionPtr f,
                                                                   constraint::Type t,
                                                                   const Eigen::VectorXd & rhs) const override;
  task_dynamics::Order order_() const override;

  TASK_DYNAMICS_DERIVED_FACTORY(a_, b_, c_)

private:
  double a_;
  double b_;
  double c_;
};
} // namespace tvm::example

// So far tvm::function::abstract::Function was only forward-declared
#include <tvm/function/abstract/Function.h>

namespace tvm::example
{
AdaptiveProportional::AdaptiveProportional(double a, double b, double c) : a_(a), b_(b), c_(c) {}

std::unique_ptr<task_dynamics::abstract::TaskDynamicsImpl> AdaptiveProportional::impl_(FunctionPtr f,
                                                                                       constraint::Type t,
                                                                                       const Eigen::VectorXd & rhs) const
{
  return std::make_unique<Impl>(f, t, rhs, a_, b_, c_);
}

task_dynamics::Order AdaptiveProportional::order_() const { return task_dynamics::Order::One; }

AdaptiveProportional::Impl::Impl(FunctionPtr f,
                                 constraint::Type t,
                                 const Eigen::VectorXd & rhs,
                                 double a,
                                 double b,
                                 double c)
: TaskDynamicsImpl(task_dynamics::Order::One, f, t, rhs), a_(a), b_(b), c_(c)
{
}

void AdaptiveProportional::Impl::updateValue()
{
  value_ = function().value() - rhs(); // e = f - rhs
  double kp = a_ * exp(-b_ * value_.norm()) + c_; // k_p = a exp(-b ||e||) + c
  value_ *= -kp; // \dot{e} = -k_p e
}
} // namespace tvm::example

// Let's run some quick tests to ensure this example is not outdated and compiles.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <tvm/function/IdentityFunction.h>

using namespace tvm;
using namespace Eigen;

TEST_CASE("AdaptiveProportional test")
{
  // Creation of a variable of size 3 and initialization
  VariablePtr x = Space(3).createVariable("x");
  x << .1, -.2, .3;

  // Creation of an identity function f and a rhs;
  auto f = std::make_shared<function::IdentityFunction>(x);
  VectorXd rhs = Vector3d(.1, .1, -.1);

  // Creating the TaskDynamics object
  example::AdaptiveProportional ap(1, 2, 0.1);

  // Getting the implementation (not usually done by the user)
  // The type is not important here
  auto impl = ap.impl(f, constraint::Type::EQUAL, rhs);

  // Usually, the computation graph is handled automatically, but here we need to
  // trigger the updates manually.
  f->updateValue();
  impl->updateValue();

  // We have e = f-rhs = (0,-.3,.4) and ||e|| = .5
  // a exp(-b ||e||) + c = exp(-1) + 0.1
  double kp = exp(-1) + 0.1;
  FAST_CHECK_UNARY(impl->value().isApprox(-kp * Vector3d(0, -.3, .4)));
}
