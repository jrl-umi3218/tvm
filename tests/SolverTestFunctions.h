

#include <tvm/function/abstract/Function.h>
#include <tvm/graph/abstract/OutputSelector.h>

using namespace tvm;
using namespace Eigen;

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

  Simple2dRobotEE(VariablePtr x, const Vector2d& base, const VectorXd& lengths);

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


/** A brken f(x) = (x-x0)^2 - r^2
  *
  */
class BrokenSphereFunction : public graph::abstract::OutputSelector<function::abstract::Function>
{
public:
  DISABLE_OUTPUTS(function::abstract::Function::Output::JDot);
  SET_UPDATES(BrokenSphereFunction, Value, Jacobian, VelocityAndAcc);

  BrokenSphereFunction(VariablePtr x, const VectorXd& x0, double radius);

  void breakJacobian(bool b);
  void breakVelocity(bool b);
  void breakNormalAcceleration(bool b);

  void updateValue();
  void updateJacobian();
  void updateVelocityAndNormalAcc();

private:
  int dimension_;
  double radius2_;
  VectorXd x0_;
  bool breakJacobian_;
  bool breakVelocity_;
  bool breakNormalAcceleration_;

};
