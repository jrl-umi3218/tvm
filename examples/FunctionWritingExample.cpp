// Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM

#include <tvm/function/abstract/Function.h>

namespace tvm::example
{
class DotProduct : public function::abstract::Function
{
public:
  SET_UPDATES(DotProduct, Value, Jacobian, VelocityAndNormalAcc, JDot)

  DotProduct(VariablePtr x, VariablePtr y);

  void updateValue();
  void updateJacobian();
  void updateVelocityAndNormalAcc();
  void updateJDot();

private:
  Variable & x_;
  Variable & y_;
  Variable & dx_;
  Variable & dy_;
};

DotProduct::DotProduct(VariablePtr x, VariablePtr y) : Function(1), x_(*x), y_(*y), dx_(*dot(x)), dy_(*dot(y))
{
  // clang-format off
    registerUpdates(Update::Value, &DotProduct::updateValue,
      Update::Jacobian, &DotProduct::updateJacobian,
      Update::VelocityAndNormalAcc, &DotProduct::updateVelocityAndNormalAcc,
      Update::JDot, &DotProduct::updateJDot);
  // clang-format on
  addOutputDependency<DotProduct>(Output::Value, Update::Value);
  addOutputDependency<DotProduct>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<DotProduct>({Output::Velocity, Output::NormalAcceleration}, Update::VelocityAndNormalAcc);
  addOutputDependency<DotProduct>(Output::JDot, Update::JDot);
  addVariable(x, true);
  addVariable(y, true);
}

void DotProduct::updateValue() { value_ = x_.value().transpose() * y_.value(); }

void DotProduct::updateJacobian()
{
  jacobian_.at(&x_) = y_.value().transpose();
  jacobian_.at(&y_) = x_.value().transpose();
}

void DotProduct::updateVelocityAndNormalAcc()
{
  velocity_ = dx_.value().transpose() * y_.value() + x_.value().transpose() * dy_.value();
  normalAcceleration_ = 2 * dx_.value().transpose() * dy_.value();
}

void DotProduct::updateJDot()
{
  JDot_.at(&x_) = dy_.value().transpose();
  JDot_.at(&y_) = dx_.value().transpose();
}
} // namespace tvm::example

#include <tvm/graph/abstract/OutputSelector.h>
#include <vector>

namespace tvm::example
{
class FunctionDotProduct : public graph::abstract::OutputSelector<function::abstract::Function>
{
public:
  DISABLE_OUTPUTS(Output::JDot)
  SET_UPDATES(FunctionDotProduct, Value, Jacobian, Velocity, NormalAcc)

  FunctionDotProduct(FunctionPtr g, FunctionPtr h);

  void updateValue();
  void updateJacobian();
  void updateVelocity();
  void updateNormalAcc();

private:
  // Register the inputs and update, and specify the dependencies output <- update <- inputs
  template<typename Out, typename Up, typename... In>
  void processOutput(Out output, Up u, void (FunctionDotProduct::*update)(), In... inputs);

  FunctionPtr g_;
  FunctionPtr h_;
};

FunctionDotProduct::FunctionDotProduct(FunctionPtr g, FunctionPtr h)
: graph::abstract::OutputSelector<function::abstract::Function>(1), g_(g), h_(h)
{
  if(!g->imageSpace().isEuclidean() || !h->imageSpace().isEuclidean())
    throw std::runtime_error("Function g and h must have their values in an Euclidean space.");
  if(g->size() != h->size())
    throw std::runtime_error("Function g and h must have the same size");

  processOutput(Output::Value, Update::Value, &FunctionDotProduct::updateValue, Output::Value);
  processOutput(Output::Jacobian, Update::Jacobian, &FunctionDotProduct::updateJacobian, Output::Value,
                Output::Jacobian);
  processOutput(Output::Velocity, Update::Velocity, &FunctionDotProduct::updateVelocity, Output::Value,
                Output::Velocity);
  processOutput(Output::NormalAcceleration, Update::NormalAcc, &FunctionDotProduct::updateNormalAcc, Output::Value,
                Output::Velocity, Output::NormalAcceleration);

  for(const auto & xi : g_->variables())
  {
    bool lin = g_->linearIn(*xi) && !h_->variables().contains(*xi);
    addVariable(xi, lin);
  }
  for(const auto & xi : h_->variables())
  {
    bool lin = h_->linearIn(*xi) && !g_->variables().contains(*xi);
    addVariable(xi, lin);
  }
}

void FunctionDotProduct::updateValue() { value_ = g_->value().transpose() * h_->value(); }

void FunctionDotProduct::updateJacobian()
{
  // We reset all the jacobian matrices (we could do that only for those corresponding to variable of h)
  for(const auto & xi : variables_)
  {
    jacobian_.at(xi.get()).setZero();
  }

  for(const auto & xi : g_->variables())
  {
    jacobian_.at(xi.get()) = h_->value().transpose() * g_->jacobian(*xi);
  }

  // We use += here, because if a variable is also present in g, we need to add to the previously copied jacobian matrix
  for(const auto & xi : h_->variables())
  {
    jacobian_.at(xi.get()) += g_->value().transpose() * h_->jacobian(*xi);
  }
}

void FunctionDotProduct::updateVelocity()
{
  velocity_ = g_->value().transpose() * h_->velocity() + g_->velocity().transpose() * h_->value();
}

void FunctionDotProduct::updateNormalAcc()
{
  normalAcceleration_ = 2 * g_->velocity().transpose() * h_->velocity()
                        + g_->value().transpose() * h_->normalAcceleration()
                        + h_->value().transpose() * g_->normalAcceleration();
}

template<typename Out, typename Up, typename... In>
void FunctionDotProduct::processOutput(Out output, Up u, void (FunctionDotProduct::*update)(), In... inputs)
{
  // We enable the output is all the required inputs are enabled for g and h
  bool enableOutput = (... && (g_->isOutputEnabled(inputs) && h_->isOutputEnabled(inputs)));

  if(enableOutput)
  {
    addInput(g_, inputs...);
    addInput(h_, inputs...);
    registerUpdates(u, update);
    addInputDependency<FunctionDotProduct>(u, g_, inputs...);
    addInputDependency<FunctionDotProduct>(u, h_, inputs...);
    addOutputDependency<FunctionDotProduct>(output, u);
  }
  else
    disableOutput(output);
}
} // namespace tvm::example

// Let's run some quick tests to ensure these examples are not outdated and compile.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <iostream>
#include <tvm/function/IdentityFunction.h>
#include <tvm/utils/graph.h>

using namespace tvm;
using namespace tvm::example;
using namespace Eigen;

TEST_CASE("DotProduct")
{
  // Creating and initializing variables
  Space R3(3);
  VariablePtr x = R3.createVariable("x");
  x << 1, 2, 3;
  dot(x) << -1, -2, -3;
  VariablePtr y = R3.createVariable("y");
  y << 4, 5, 6;
  dot(y) << -4, -5, -6;

  // Creating function
  DotProduct dp(x, y);

  // If we do not call the update methods, the values will be wrong
  double val = dp.value()[0];
  MatrixXd Jx = dp.jacobian(*x);
  FAST_CHECK_NE(val, 32);
  FAST_CHECK_UNARY_FALSE(Jx.isApprox(y->value().transpose()));

  // Manually updating the values of dp
  dp.updateValue();
  dp.updateJacobian();
  dp.updateVelocityAndNormalAcc();
  dp.updateJDot();

  // Check the results
  val = dp.value()[0];
  Jx = dp.jacobian(*x);
  MatrixXd Jy = dp.jacobian(*y);
  double vel = dp.velocity()[0];
  double na = dp.normalAcceleration()[0];
  MatrixXd Jdx = dp.JDot(*x);
  MatrixXd Jdy = dp.JDot(*y);
  FAST_CHECK_EQ(val, 32);
  FAST_CHECK_UNARY(Jx.isApprox(y->value().transpose()));
  FAST_CHECK_UNARY(Jy.isApprox(x->value().transpose()));
  FAST_CHECK_EQ(vel, -64);
  FAST_CHECK_EQ(na, 64);
  FAST_CHECK_UNARY(Jdx.isApprox(dot(y)->value().transpose()));
  FAST_CHECK_UNARY(Jdy.isApprox(dot(x)->value().transpose()));
}

// An identity function whose Velocity output was disabled
class DummyFunction : public function::IdentityFunction
{
public:
  DISABLE_OUTPUTS(Output::Velocity)

  DummyFunction(VariablePtr x) : IdentityFunction(x) {}
};

TEST_CASE("FunctionDotProduct")
{
  // Creating and initializing variables
  Space R3(3);
  VariablePtr x = R3.createVariable("x");
  x << 1, 2, 3;
  dot(x) << -1, -2, -3;
  VariablePtr y = R3.createVariable("y");
  y << 4, 5, 6;
  dot(y) << -4, -5, -6;

  // Function x + y
  auto sum = std::make_shared<function::BasicLinearFunction>(x + y);
  // Function y
  auto dum = std::make_shared<DummyFunction>(x);

  // Function y^T (x + y)
  auto dotf = std::make_shared<FunctionDotProduct>(sum, dum);

  // Testing availability of outputs
  FAST_CHECK_UNARY(dotf->isOutputEnabled(FunctionDotProduct::Output::Value));
  FAST_CHECK_UNARY(dotf->isOutputEnabled(FunctionDotProduct::Output::Jacobian));
  FAST_CHECK_UNARY_FALSE(dotf->isOutputEnabled(FunctionDotProduct::Output::Velocity));
  FAST_CHECK_UNARY_FALSE(dotf->isOutputEnabled(FunctionDotProduct::Output::NormalAcceleration));
  FAST_CHECK_UNARY_FALSE(dotf->isOutputEnabled(FunctionDotProduct::Output::JDot));

  // Testing linearity
  FAST_CHECK_UNARY_FALSE(dotf->linearIn(*x));
  FAST_CHECK_UNARY(dotf->linearIn(*y));

  // Using utility function to generate the computation graph and execute it
  auto gl = utils::generateUpdateGraph(dotf, FunctionDotProduct::Output::Value, FunctionDotProduct::Output::Jacobian);
  gl->execute();

  // Testing the results
  double val = dotf->value()[0];
  MatrixXd Jx = dotf->jacobian(*x);
  MatrixXd Jy = dotf->jacobian(*y);
  FAST_CHECK_EQ(val, 46);
  FAST_CHECK_UNARY(Jx.isApprox((2 * x->value() + y->value()).transpose()));
  FAST_CHECK_UNARY(Jy.isApprox(x->value().transpose()));
}
