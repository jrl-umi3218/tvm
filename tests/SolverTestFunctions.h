/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/function/abstract/Function.h>
#include <tvm/graph/abstract/OutputSelector.h>

/** f(x) = (x-x0)^2 - r^2*/
class SphereFunction : public tvm::graph::abstract::OutputSelector<tvm::function::abstract::Function>
{
public:
  DISABLE_OUTPUTS(Output::JDot)
  SET_UPDATES(SphereFunction, Value, Jacobian, VelocityAndAcc)

  SphereFunction(tvm::VariablePtr x, const Eigen::VectorXd & x0, double radius);

  void updateValue();
  void updateJacobian();
  void updateVelocityAndNormalAcc();

private:
  int dimension_;
  double radius2_;
  Eigen::VectorXd x0_;
};

class Simple2dRobotEE : public tvm::graph::abstract::OutputSelector<tvm::function::abstract::Function>
{
public:
  DISABLE_OUTPUTS(Output::JDot)
  SET_UPDATES(Simple2dRobotEE, Value, Jacobian, VelocityAndAcc)

  Simple2dRobotEE(tvm::VariablePtr x, const Eigen::Vector2d & base, const Eigen::VectorXd & lengths);

  void updateValue();
  void updateJacobian();
  void updateVelocityAndNormalAcc();

private:
  int n_;
  Eigen::Vector2d base_;
  Eigen::VectorXd lengths_;
};

// f - g
class Difference : public tvm::graph::abstract::OutputSelector<tvm::function::abstract::Function>
{
public:
  SET_UPDATES(Difference, Value, Jacobian, Velocity, NormalAcceleration, JDot)

  Difference(tvm::FunctionPtr f, tvm::FunctionPtr g);

  void updateValue();
  void updateJacobian();
  void updateVelocity();
  void updateNormalAcceleration();
  void updateJDot();

private:
  template<typename Input>
  void processOutput(Input input, Update_ u, void (Difference::*update)());

  tvm::FunctionPtr f_;
  tvm::FunctionPtr g_;
};

/** A broken f(x) = (x-x0)^2 - r^2
 *
 */
class BrokenSphereFunction : public tvm::graph::abstract::OutputSelector<tvm::function::abstract::Function>
{
public:
  DISABLE_OUTPUTS(Output::JDot)
  SET_UPDATES(BrokenSphereFunction, Value, Jacobian, VelocityAndAcc)

  BrokenSphereFunction(tvm::VariablePtr x, const Eigen::VectorXd & x0, double radius);

  void breakJacobian(bool b);
  void breakVelocity(bool b);
  void breakNormalAcceleration(bool b);

  void updateValue();
  void updateJacobian();
  void updateVelocityAndNormalAcc();

private:
  int dimension_;
  double radius2_;
  Eigen::VectorXd x0_;
  bool breakJacobian_;
  bool breakVelocity_;
  bool breakNormalAcceleration_;
};
