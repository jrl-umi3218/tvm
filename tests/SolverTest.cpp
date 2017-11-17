#include <iostream>

#include <tvm/ControlProblem.h>
#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/constraint/abstract/Constraint.h>
#include <tvm/function/abstract/Function.h>
#include <tvm/function/abstract/LinearFunction.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/graph/abstract/OutputSelector.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>

using namespace tvm;
using namespace Eigen;
using namespace tvm::utils;

/** f(x) = (x-x0)^2 - r^2*/
class SphereFunction : public graph::abstract::OutputSelector<function::abstract::Function>
{
public:
  DISABLE_OUTPUTS(function::abstract::Function::Output::JDot);
  SET_UPDATES(SphereFunction, Value, Jacobian, VelocityAndAcc);

  SphereFunction(VariablePtr x, const VectorXd& x0, double radius);

  void updateValue();
  void updateJacobian();
  void updateVelocityAndNormalAcc();

private:
  int dimension_;
  double radius2_;
  VectorXd x0_;
};



class Simple2dRobotEE : public graph::abstract::OutputSelector<function::abstract::Function>
{
public:
  DISABLE_OUTPUTS(function::abstract::Function::Output::JDot);
  SET_UPDATES(Simple2dRobotEE, Value, Jacobian, VelocityAndAcc);

  Simple2dRobotEE(VariablePtr x,  const Vector2d& base, const VectorXd& lengths);

  void updateValue();
  void updateJacobian();
  void updateVelocityAndNormalAcc();

private:
  int n_;
  Vector2d base_;
  VectorXd lengths_;
};


//f - g
class Difference : public graph::abstract::OutputSelector<function::abstract::Function>
{
public:
  SET_UPDATES(Difference, Value, Jacobian, Velocity, NormalAcceleration, JDot);

  Difference(FunctionPtr f, FunctionPtr g);

  void updateValue();
  void updateJacobian();
  void updateVelocity();
  void updateNormalAcceleration();
  void updateJDot();

private:
  template<typename Input>
  void processOutput(Input input, Update_ u, void (Difference::*update)());

  FunctionPtr f_;
  FunctionPtr g_;
};


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
  jacobian_.begin()->second = 2 * variables()[0]->value().transpose();
}

void SphereFunction::updateVelocityAndNormalAcc()
{
  const auto& x = variables()[0]->value();
  const auto& v = dot(variables()[0])->value();

  velocity_[0] = 2*x.dot(v);
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



void checkJacobian(FunctionPtr f)
{
  auto userValue = std::make_shared<graph::internal::Inputs>();
  userValue->addInput(f, tvm::internal::FirstOrderProvider::Output::Value);
  auto userFull = std::make_shared<graph::internal::Inputs>();
  userFull->addInput(f, tvm::internal::FirstOrderProvider::Output::Value);
  userFull->addInput(f, tvm::internal::FirstOrderProvider::Output::Jacobian);

  tvm::graph::CallGraph gValue;
  gValue.add(userValue);
  gValue.update();
  tvm::graph::CallGraph gFull;
  gFull.add(userFull);
  gFull.update();

  for (auto x : f->variables())
  {
    VectorXd x0 = VectorXd::Random(x->size());
    x->value(x0);
    gFull.execute();
    auto f0 = f->value();
    auto J0 = f->jacobian(*x);

    const double h = 1e-6;
    MatrixXd J(f->size(), x->size());
    for (int i = 0; i < x->size(); ++i)
    {
      auto xi = x0;
      xi[i] += h;
      x->value(xi);
      gValue.execute();
      J.col(i) = (f->value() - f0) / h;
    }

    std::cout << x->name() << ": " << (J - J0).array().abs().maxCoeff() << std::endl;
  }
}

void checkNormalAcc(FunctionPtr f)
{
  auto user1stOrder = std::make_shared<graph::internal::Inputs>();
  user1stOrder->addInput(f, tvm::internal::FirstOrderProvider::Output::Value);
  user1stOrder->addInput(f, tvm::internal::FirstOrderProvider::Output::Jacobian);
  auto userFull = std::make_shared<graph::internal::Inputs>();
  userFull->addInput(f, tvm::internal::FirstOrderProvider::Output::Value);
  userFull->addInput(f, tvm::internal::FirstOrderProvider::Output::Jacobian);
  userFull->addInput(f, function::abstract::Function::Output::Velocity);
  userFull->addInput(f, function::abstract::Function::Output::NormalAcceleration);

  tvm::graph::CallGraph g1stOrder;
  g1stOrder.add(user1stOrder);
  g1stOrder.update();
  tvm::graph::CallGraph gFull;
  gFull.add(userFull);
  gFull.update();

  VectorXd na = VectorXd::Zero(f->size());
  VectorXd na0;
  for (auto x : f->variables())
  {
    VariablePtr v = dot(x);
    VectorXd x0 = VectorXd::Random(x->size());
    VectorXd v0 = VectorXd::Random(x->size());
    x->value(x0);
    v->value(v0);
    gFull.execute();
    auto J0 = f->jacobian(*x);
    na0 = f->normalAcceleration();

    const double h = 1e-6;
    x->value(x0 + h*v0);
    g1stOrder.execute();
    MatrixXd J1 = f->jacobian(*x);
    na += (J1 - J0) / h * v0;
  }

  std::cout << (na - na0).array().abs().maxCoeff() << std::endl;
}


void solverTest01()
{
  Space s1(2);
  VariablePtr x = s1.createVariable("x");

  Space s2(3);
  VariablePtr q = s2.createVariable("q");

  auto sf = std::make_shared<SphereFunction>(x, Vector2d(0, 0), 1);
  auto rf = std::make_shared<Simple2dRobotEE>(q, Vector2d(2, 0), Vector3d(1, 1, 1));
  auto idx = std::make_shared<function::IdentityFunction>(x);
  auto df = std::make_shared<Difference>(rf, idx);

  checkJacobian(sf);
  checkNormalAcc(sf);
  checkJacobian(rf);
  checkNormalAcc(rf);
  checkJacobian(idx);
  checkNormalAcc(idx);
  checkJacobian(df);
  checkNormalAcc(df);

  ControlProblem pb;
  pb.add(sf == 0., task_dynamics::ProportionalDerivative(2), { requirements::PriorityLevel(0) });
  pb.add(df == 0., task_dynamics::PD(2), { requirements::PriorityLevel(0) });

  LinearizedControlProblem lpb(pb);

  scheme::WeightedLeastSquares solver;
  solver.solve(lpb);
  std::cout << "ddx = " << dot(x, 2)->value().transpose() << std::endl;
  std::cout << "ddq = " << dot(q, 2)->value().transpose() << std::endl;
}
