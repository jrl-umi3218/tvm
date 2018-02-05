#include "SolverTestFunctions.h"
#include <tvm/Variable.h>

SphereFunction::SphereFunction(VariablePtr x, const VectorXd & x0, double radius)
  : graph::abstract::OutputSelector<function::abstract::Function>(1)
  , dimension_(x->size())
  , radius2_(radius*radius)
  , x0_(x0)
{
  assert(x->size() == x0.size());

  registerUpdates(SphereFunction::Update::Value, &SphereFunction::updateValue);
  registerUpdates(SphereFunction::Update::Jacobian, &SphereFunction::updateJacobian);
  registerUpdates(SphereFunction::Update::VelocityAndAcc, &SphereFunction::updateVelocityAndNormalAcc);
  addOutputDependency<SphereFunction>(FirstOrderProvider::Output::Value, SphereFunction::Update::Value);
  addOutputDependency<SphereFunction>(FirstOrderProvider::Output::Jacobian, SphereFunction::Update::Jacobian);
  addOutputDependency<SphereFunction>({ function::abstract::Function::Output::Velocity, function::abstract::Function::Output::NormalAcceleration }, SphereFunction::Update::VelocityAndAcc);

  addVariable(x, false);
}

void SphereFunction::updateValue()
{
  value_[0] = (variables()[0]->value() - x0_).squaredNorm() - radius2_;
}

void SphereFunction::updateJacobian()
{
  jacobian_.begin()->second = 2 * (variables()[0]->value() - x0_).transpose();
}

void SphereFunction::updateVelocityAndNormalAcc()
{
  const auto& x = variables()[0]->value();
  const auto& v = dot(variables()[0])->value();

  velocity_[0] = 2*(x-x0_).dot(v);
  normalAcceleration_[0] = 2 * v.squaredNorm();
}

Matrix3d H(double t, double l)
{
  double c = std::cos(t);
  double s = std::sin(t);
  return (Matrix3d() << c, -s, c*l, s, c, s*l, 0, 0, 1).finished();
}

//dH/dt
Matrix3d dH(double t, double l)
{
  double c = std::cos(t);
  double s = std::sin(t);
  return (Matrix3d() << -s, -c, -s*l, c, -s, c*l, 0, 0, 0).finished();
}

//d^2H/dt^2
Matrix3d ddH(double t, double l)
{
  double c = std::cos(t);
  double s = std::sin(t);
  return (Matrix3d() << -c, s, -c*l, -s, -c, -s*l, 0, 0, 0).finished();
}

Simple2dRobotEE::Simple2dRobotEE(VariablePtr x, const Vector2d& base, const VectorXd& lengths)
  : graph::abstract::OutputSelector<function::abstract::Function>(2)
  , n_(x->size())
  , base_(base)
  , lengths_(lengths)
{
  assert(x->size() == lengths.size());

  registerUpdates(Simple2dRobotEE::Update::Value, &Simple2dRobotEE::updateValue);
  registerUpdates(Simple2dRobotEE::Update::Jacobian, &Simple2dRobotEE::updateJacobian);
  registerUpdates(Simple2dRobotEE::Update::VelocityAndAcc, &Simple2dRobotEE::updateVelocityAndNormalAcc);
  addOutputDependency<Simple2dRobotEE>(FirstOrderProvider::Output::Value, Simple2dRobotEE::Update::Value);
  addOutputDependency<Simple2dRobotEE>(FirstOrderProvider::Output::Jacobian, Simple2dRobotEE::Update::Jacobian);
  addOutputDependency<Simple2dRobotEE>({ function::abstract::Function::Output::Velocity, function::abstract::Function::Output::NormalAcceleration }, Simple2dRobotEE::Update::VelocityAndAcc);
  addInternalDependency<Simple2dRobotEE>(Simple2dRobotEE::Update::VelocityAndAcc, Simple2dRobotEE::Update::Jacobian);

  addVariable(x, false);
}

void Simple2dRobotEE::updateValue()
{
  const auto& x = variables()[0]->value();

  Vector3d v(0, 0, 1);
  for (int i = n_-1; i >= 0; --i)
    v = H(x[i], lengths_[i]) * v;
  value_ = v.head<2>();
}

void Simple2dRobotEE::updateJacobian()
{
  //a very inefficient way to compute the jacobian
  const auto& x = variables()[0]->value();

  for (int j = 0; j < n_; ++j)
  {
    Vector3d v(0,0,1);

    for (int i = n_-1; i >= j+1; --i)
      v = H(x[i], lengths_[i]) * v;

    v = dH(x[j], lengths_[j]) * v;

    for (int i=j-1; i>=0; --i)
      v = H(x[i], lengths_[i]) * v;

    jacobian_.begin()->second.col(j) = v.head<2>();
  }
}

void Simple2dRobotEE::updateVelocityAndNormalAcc()
{
  const auto& v = dot(variables()[0])->value();
  velocity_ = jacobian_.begin()->second * v;

  //even more inefficient way to compute Jdotqdot
  //we use the fact that \dot{J} = sum \dot{q}_k dJ/dq_k
  const auto& x = variables()[0]->value();

  normalAcceleration_.setZero();
  for (int k = 0; k < n_; ++k)
  {
    //computation of dJ/dq_k, column, by column
    MatrixXd dJ = MatrixXd::Zero(2, n_);
    for (int j = 0; j < n_; ++j)
    {
      Vector3d v(0,0,1);
      if (j==k)
      {
        for (int i = n_-1; i >= j+1; --i)
          v = H(x[i], lengths_[i]) * v;

        v = ddH(x[j], lengths_[j]) * v;

        for (int i = j-1; i >= 0; --i)
          v = H(x[i], lengths_[i]) * v;
      }
      else
      {
        auto l = std::minmax(j, k);
        for (int i = n_-1; i >= l.second+1; --i)
          v = H(x[i], lengths_[i]) * v;

        v = dH(x[l.second], lengths_[l.second]) * v;

        for (int i = l.second- 1; i >= l.first+1; --i)
          v = H(x[i], lengths_[i]) * v;

        v = dH(x[l.first], lengths_[l.first]) * v;

        for (int i = l.first-1; i >= 0; --i)
          v = H(x[i], lengths_[i]) * v;
      }

      dJ.col(j) = v.head<2>();
    }

    //\dot{J}\dot{q} = sum \dot{q}_k (dJ/dq_k*\dot{q})
    normalAcceleration_ += v[k] * (dJ*v);
  }
}


Difference::Difference(FunctionPtr f, FunctionPtr g)
  :graph::abstract::OutputSelector<function::abstract::Function>(f->size())
  , f_(f)
  , g_(g)
{
  assert(f->size() == g->size());

  using BasicOutput = tvm::internal::FirstOrderProvider::Output;
  using AdvancedOutput = function::abstract::Function::Output;
  processOutput(BasicOutput::Value, Update::Value, &Difference::updateValue);
  processOutput(BasicOutput::Jacobian, Update::Jacobian, &Difference::updateJacobian);
  processOutput(AdvancedOutput::Velocity, Update::Velocity, &Difference::updateVelocity);
  processOutput(AdvancedOutput::NormalAcceleration, Update::NormalAcceleration, &Difference::updateNormalAcceleration);
  processOutput(AdvancedOutput::JDot, Update::JDot, &Difference::updateJDot);

  //Note: if we copy this class later, we need to take care of the linearity of each variable properly
  const auto& fvars = f->variables();
  for (const auto& v : fvars)
    addVariable(v, false);

  for (const auto& v : g->variables())
  {
    if (std::find(fvars.begin(), fvars.end(), v) == fvars.end())
      addVariable(v, false);
  }
}

template<typename Input>
void Difference::processOutput(Input input, Difference::Update_ u, void (Difference::*update)())
{
  if (f_->isOutputEnabled(input) && g_->isOutputEnabled(input))
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

void Difference::updateValue()
{
  value_ = f_->value() - g_->value();
}

void Difference::updateJacobian()
{
  for (const auto& v : g_->variables())
    jacobian_.at(v.get()).setZero();

  for (const auto& v : f_->variables())
    jacobian_.at(v.get()) = f_->jacobian(*v);

  for (const auto& v : g_->variables())
    jacobian_.at(v.get()) -= g_->jacobian(*v);
}

void Difference::updateVelocity()
{
  velocity_ = f_->velocity() - g_->velocity();
}

void Difference::updateNormalAcceleration()
{
  normalAcceleration_ = f_->normalAcceleration() - g_->normalAcceleration();
}

void Difference::updateJDot()
{
  for (const auto& v : g_->variables())
    JDot_.at(v.get()).setZero();

  for (const auto& v : f_->variables())
    JDot_.at(v.get()) = f_->JDot(*v);

  for (const auto& v : g_->variables())
    JDot_.at(v.get()) -= g_->JDot(*v);
}

BrokenSphereFunction::BrokenSphereFunction(VariablePtr x, const VectorXd & x0, double radius)
  : graph::abstract::OutputSelector<function::abstract::Function>(1)
  , dimension_(x->size())
  , radius2_(radius*radius)
  , x0_(x0)
  , breakJacobian_(false)
  , breakVelocity_(false)
  , breakNormalAcceleration_(false)
{
  assert(x->size() == x0.size());

  registerUpdates(BrokenSphereFunction::Update::Value, &BrokenSphereFunction::updateValue);
  registerUpdates(BrokenSphereFunction::Update::Jacobian, &BrokenSphereFunction::updateJacobian);
  registerUpdates(BrokenSphereFunction::Update::VelocityAndAcc, &BrokenSphereFunction::updateVelocityAndNormalAcc);
  addOutputDependency<BrokenSphereFunction>(FirstOrderProvider::Output::Value, BrokenSphereFunction::Update::Value);
  addOutputDependency<BrokenSphereFunction>(FirstOrderProvider::Output::Jacobian, BrokenSphereFunction::Update::Jacobian);
  addOutputDependency<BrokenSphereFunction>({ function::abstract::Function::Output::Velocity, function::abstract::Function::Output::NormalAcceleration }, BrokenSphereFunction::Update::VelocityAndAcc);

  addVariable(x, false);
}

void BrokenSphereFunction::breakJacobian(bool b)
{
  breakJacobian_ = b;
}

void BrokenSphereFunction::breakVelocity(bool b)
{
  breakVelocity_ = b;
}

void BrokenSphereFunction::breakNormalAcceleration(bool b)
{
  breakNormalAcceleration_ = b;
}

void BrokenSphereFunction::updateValue()
{
  value_[0] = (variables()[0]->value() - x0_).squaredNorm() - radius2_;
}

void BrokenSphereFunction::updateJacobian()
{
  if (breakJacobian_)
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
  const auto& x = variables()[0]->value();
  const auto& v = dot(variables()[0])->value();

  if (breakVelocity_)
  {
    velocity_[0] = 2 * (x).dot(v + x0_);
  }
  else
  {
    velocity_[0] = 2 * (x - x0_).dot(v);
  }
  if (breakNormalAcceleration_)
  {
    normalAcceleration_[0] = 2 * v.dot(x0_);
  }
  else
  {
    normalAcceleration_[0] = 2 * v.squaredNorm();
  }
}
