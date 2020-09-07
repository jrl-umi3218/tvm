/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include "SolverTestFunctions.h"
#include <tvm/Variable.h>

using namespace Eigen;
using namespace tvm;

SphereFunction::SphereFunction(VariablePtr x, const VectorXd & x0, double radius)
: graph::abstract::OutputSelector<function::abstract::Function>(1), dimension_(x->size()), radius2_(radius * radius),
  x0_(x0)
{
  assert(x->size() == x0.size());

  registerUpdates(Update::Value, &SphereFunction::updateValue);
  registerUpdates(Update::Jacobian, &SphereFunction::updateJacobian);
  registerUpdates(Update::VelocityAndAcc, &SphereFunction::updateVelocityAndNormalAcc);
  addOutputDependency<SphereFunction>(Output::Value, Update::Value);
  addOutputDependency<SphereFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<SphereFunction>({Output::Velocity, Output::NormalAcceleration}, Update::VelocityAndAcc);

  addVariable(x, false);
}

void SphereFunction::updateValue() { value_[0] = (variables()[0]->value() - x0_).squaredNorm() - radius2_; }

void SphereFunction::updateJacobian() { jacobian_.begin()->second = 2 * (variables()[0]->value() - x0_).transpose(); }

void SphereFunction::updateVelocityAndNormalAcc()
{
  const auto & x = variables()[0]->value();
  const auto & v = dot(variables()[0])->value();

  velocity_[0] = 2 * (x - x0_).dot(v);
  normalAcceleration_[0] = 2 * v.squaredNorm();
}

Matrix3d H(double t, double l)
{
  double c = std::cos(t);
  double s = std::sin(t);
  return (Matrix3d() << c, -s, c * l, s, c, s * l, 0, 0, 1).finished();
}

// dH/dt
Matrix3d dH(double t, double l)
{
  double c = std::cos(t);
  double s = std::sin(t);
  return (Matrix3d() << -s, -c, -s * l, c, -s, c * l, 0, 0, 0).finished();
}

// d^2H/dt^2
Matrix3d ddH(double t, double l)
{
  double c = std::cos(t);
  double s = std::sin(t);
  return (Matrix3d() << -c, s, -c * l, -s, -c, -s * l, 0, 0, 0).finished();
}

Simple2dRobotEE::Simple2dRobotEE(VariablePtr x, const Vector2d & base, const VectorXd & lengths)
: graph::abstract::OutputSelector<function::abstract::Function>(2), n_(x->size()), base_(base), lengths_(lengths)
{
  assert(x->size() == lengths.size());

  registerUpdates(Update::Value, &Simple2dRobotEE::updateValue);
  registerUpdates(Update::Jacobian, &Simple2dRobotEE::updateJacobian);
  registerUpdates(Update::VelocityAndAcc, &Simple2dRobotEE::updateVelocityAndNormalAcc);
  addOutputDependency<Simple2dRobotEE>(Output::Value, Update::Value);
  addOutputDependency<Simple2dRobotEE>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<Simple2dRobotEE>({Output::Velocity, Output::NormalAcceleration}, Update::VelocityAndAcc);
  addInternalDependency<Simple2dRobotEE>(Update::VelocityAndAcc, Update::Jacobian);

  addVariable(x, false);
}

void Simple2dRobotEE::updateValue()
{
  const auto & x = variables()[0]->value();

  Vector3d v(0, 0, 1);
  for(int i = n_ - 1; i >= 0; --i)
    v = H(x[i], lengths_[i]) * v;
  value_ = v.head<2>() + base_;
}

void Simple2dRobotEE::updateJacobian()
{
  // a very inefficient way to compute the jacobian
  const auto & x = variables()[0]->value();

  for(int j = 0; j < n_; ++j)
  {
    Vector3d v(0, 0, 1);

    for(int i = n_ - 1; i >= j + 1; --i)
      v = H(x[i], lengths_[i]) * v;

    v = dH(x[j], lengths_[j]) * v;

    for(int i = j - 1; i >= 0; --i)
      v = H(x[i], lengths_[i]) * v;

    jacobian_.begin()->second.col(j) = v.head<2>();
  }
}

void Simple2dRobotEE::updateVelocityAndNormalAcc()
{
  const auto & v = dot(variables()[0])->value();
  velocity_ = jacobian_.begin()->second * v;

  // even more inefficient way to compute Jdotqdot
  // we use the fact that \dot{J} = sum \dot{q}_k dJ/dq_k
  const auto & x = variables()[0]->value();

  normalAcceleration_.setZero();
  for(int k = 0; k < n_; ++k)
  {
    // computation of dJ/dq_k, column, by column
    MatrixXd dJ = MatrixXd::Zero(2, n_);
    for(int j = 0; j < n_; ++j)
    {
      Vector3d v(0, 0, 1);
      if(j == k)
      {
        for(int i = n_ - 1; i >= j + 1; --i)
          v = H(x[i], lengths_[i]) * v;

        v = ddH(x[j], lengths_[j]) * v;

        for(int i = j - 1; i >= 0; --i)
          v = H(x[i], lengths_[i]) * v;
      }
      else
      {
        auto l = std::minmax(j, k);
        for(int i = n_ - 1; i >= l.second + 1; --i)
          v = H(x[i], lengths_[i]) * v;

        v = dH(x[l.second], lengths_[l.second]) * v;

        for(int i = l.second - 1; i >= l.first + 1; --i)
          v = H(x[i], lengths_[i]) * v;

        v = dH(x[l.first], lengths_[l.first]) * v;

        for(int i = l.first - 1; i >= 0; --i)
          v = H(x[i], lengths_[i]) * v;
      }

      dJ.col(j) = v.head<2>();
    }

    //\dot{J}\dot{q} = sum \dot{q}_k (dJ/dq_k*\dot{q})
    normalAcceleration_ += v[k] * (dJ * v);
  }
}

Difference::Difference(FunctionPtr f, FunctionPtr g)
: graph::abstract::OutputSelector<function::abstract::Function>(f->size()), f_(f), g_(g)
{
  assert(f->imageSpace().isEuclidean() && g->imageSpace().isEuclidean());
  assert(f->size() == g->size());

  using BasicOutput = tvm::internal::FirstOrderProvider::Output;
  using AdvancedOutput = function::abstract::Function::Output;
  processOutput(BasicOutput::Value, Update::Value, &Difference::updateValue);
  processOutput(BasicOutput::Jacobian, Update::Jacobian, &Difference::updateJacobian);
  processOutput(AdvancedOutput::Velocity, Update::Velocity, &Difference::updateVelocity);
  processOutput(AdvancedOutput::NormalAcceleration, Update::NormalAcceleration, &Difference::updateNormalAcceleration);
  processOutput(AdvancedOutput::JDot, Update::JDot, &Difference::updateJDot);

  const auto & fvars = f->variables();
  const auto & gvars = g->variables();
  for(const auto & v : fvars)
  {
    bool lin = f->linearIn(*v);
    auto p = f->jacobian(*v).properties();
    if(gvars.contains(*v))
    {
      lin = lin && g->linearIn(*v);
      p = p - g->jacobian(*v).properties();
    }
    addVariable(v, lin);
    jacobian_.at(v.get()).properties(p);
  }

  for(const auto & v : g->variables())
  {
    if(!fvars.contains(*v))
    {
      addVariable(v, g->linearIn(*v));
      jacobian_.at(v.get()).properties(-g->jacobian(*v).properties());
    }
  }
}

template<typename Input>
void Difference::processOutput(Input input, Difference::Update_ u, void (Difference::*update)())
{
  if(f_->isOutputEnabled(input) && g_->isOutputEnabled(input))
  {
    addInput(f_, input);
    addInput(g_, input);
    registerUpdates(u, update);
    addInputDependency<Difference>(u, f_, input);
    addInputDependency<Difference>(u, g_, input);
    addOutputDependency<Difference>(input, u);
  }
  else
    disableOutput(input);
}

void Difference::updateValue() { value_ = f_->value() - g_->value(); }

void Difference::updateJacobian()
{
  for(const auto & v : g_->variables())
    jacobian_.at(v.get()).setZero();

  for(const auto & v : f_->variables())
    jacobian_.at(v.get()).keepProperties(true) = f_->jacobian(*v).matrix();

  for(const auto & v : g_->variables())
    jacobian_.at(v.get()).keepProperties(true) = jacobian_.at(v.get()) - g_->jacobian(*v);
}

void Difference::updateVelocity() { velocity_ = f_->velocity() - g_->velocity(); }

void Difference::updateNormalAcceleration()
{
  normalAcceleration_ = f_->normalAcceleration() - g_->normalAcceleration();
}

void Difference::updateJDot()
{
  for(const auto & v : g_->variables())
    JDot_.at(v.get()).setZero();

  for(const auto & v : f_->variables())
    JDot_.at(v.get()) = f_->JDot(*v);

  for(const auto & v : g_->variables())
    JDot_.at(v.get()) -= g_->JDot(*v);
}

BrokenSphereFunction::BrokenSphereFunction(VariablePtr x, const VectorXd & x0, double radius)
: graph::abstract::OutputSelector<function::abstract::Function>(1), dimension_(x->size()), radius2_(radius * radius),
  x0_(x0), breakJacobian_(false), breakVelocity_(false), breakNormalAcceleration_(false)
{
  assert(x->size() == x0.size());

  registerUpdates(Update::Value, &BrokenSphereFunction::updateValue);
  registerUpdates(Update::Jacobian, &BrokenSphereFunction::updateJacobian);
  registerUpdates(Update::VelocityAndAcc, &BrokenSphereFunction::updateVelocityAndNormalAcc);
  addOutputDependency<BrokenSphereFunction>(Output::Value, Update::Value);
  addOutputDependency<BrokenSphereFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<BrokenSphereFunction>({Output::Velocity, Output::NormalAcceleration}, Update::VelocityAndAcc);

  addVariable(x, false);
}

void BrokenSphereFunction::breakJacobian(bool b) { breakJacobian_ = b; }

void BrokenSphereFunction::breakVelocity(bool b) { breakVelocity_ = b; }

void BrokenSphereFunction::breakNormalAcceleration(bool b) { breakNormalAcceleration_ = b; }

void BrokenSphereFunction::updateValue() { value_[0] = (variables()[0]->value() - x0_).squaredNorm() - radius2_; }

void BrokenSphereFunction::updateJacobian()
{
  if(breakJacobian_)
  {
    jacobian_.begin()->second = -2 * (variables()[0]->value() - x0_).transpose();
  }
  else
  {
    jacobian_.begin()->second = 2 * (variables()[0]->value() - x0_).transpose();
  }
}

void BrokenSphereFunction::updateVelocityAndNormalAcc()
{
  const auto & x = variables()[0]->value();
  const auto & v = dot(variables()[0])->value();

  if(breakVelocity_)
  {
    velocity_[0] = 2 * (x).dot(v + x0_);
  }
  else
  {
    velocity_[0] = 2 * (x - x0_).dot(v);
  }
  if(breakNormalAcceleration_)
  {
    normalAcceleration_[0] = 2 * v.dot(x0_);
  }
  else
  {
    normalAcceleration_[0] = 2 * v.squaredNorm();
  }
}
