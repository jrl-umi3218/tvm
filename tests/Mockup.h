#include "DataGraph.h"

#include <iostream>

template <int ReturnType>
class Dummy 
{
public:
  Dummy(int i) :i_(i) {}
  operator int() { return i_; }
private:
  int i_;
};

class VariableMockup
{
public:
  enum Output {Value};

  VariableMockup() {}
  void setValue(double v) {}
  double value() const  { return 0; }
};

class RobotMockup : public taskvm::DataNode
{
public:
  enum Update { Kinematics, Velocity, Dynamics, Acceleration };
  enum Ouput { K1, K2, K3, V1, V2, D1, D2, A1 };

  RobotMockup();

  Dummy<K1> getK1() const { return k; }
  Dummy<K2> getK2() const { return k; }
  Dummy<K3> getK3() const { return k; }
  Dummy<V1> getV1() const { return v; }
  Dummy<V2> getV2() const { return v; }
  Dummy<D1> getD1() const { return d; }
  Dummy<D2> getD2() const { return d; }
  Dummy<A1> getA1() const { return a; }

protected:
  void update_(const taskvm::internal::UnifiedEnumValue& u);
  void fillOutputDependencies();
  void fillInternalDependencies();
  void fillUpdateDependencies();

private:
  void updateK() { std::cout << "update robot kinematics" << std::endl; ++k; }
  void updateV() { std::cout << "update robot velocity" << std::endl; ++v; }
  void updateD() { std::cout << "update robot dynamics" << std::endl; ++d; }
  void updateA() { std::cout << "update robot acceleration" << std::endl; ++a; }

  VariableMockup q;
  int k, v, d, a;
};

/** Base class for a time-dependent function*/
class FunctionMockup : public taskvm::DataNode
{
public:
  enum class Update {Value, Velocity, JDot};
  enum class Output {Value, Jacobian, Velocity, NormalAcceleration, JDot};

  Dummy<int(Output::Value)> value() const { return val; }
  Dummy<int(Output::Value)> jacobian() const { return j; }
  Dummy<int(Output::Value)> velocity() const { return vel; }
  Dummy<int(Output::Value)> normalAcc() const { return na; }
  Dummy<int(Output::Value)> Jdot() const { return jdot; }

protected:
  FunctionMockup(std::initializer_list<Output> outputs, std::initializer_list<Update> updates);

  int val, j, vel, na, jdot;

};

/** Base class for a function relying on a robot to compute its own outputs*/
class RobotFunction : public FunctionMockup
{
protected:
  RobotFunction(std::initializer_list<Output> outputs, std::shared_ptr<RobotMockup> robot);

protected:
  void fillInternalDependencies();
  void fillUpdateDependencies();

  std::shared_ptr<RobotMockup> robot_;
};

class SomeRobotFunction1 : public RobotFunction
{
public:
  SomeRobotFunction1(std::shared_ptr<RobotMockup> robot);

protected:
  void update_(const taskvm::internal::UnifiedEnumValue& u);
  void fillOutputDependencies();
};

class SomeRobotFunction2 : public RobotFunction
{
public:
  SomeRobotFunction2(std::shared_ptr<RobotMockup> robot);

protected:
  void update_(const taskvm::internal::UnifiedEnumValue& u);
  void fillOutputDependencies();
};

/** Base class for a linear expression Ax+b*/
class LinearConstraint: public taskvm::DataNode
{
public:
  enum class Output { Value, A, b };
  enum class Update { Matrices };

  LinearConstraint(const std::string& name);

  //example of method with argument
  Dummy<int(Output::Value)> value(int x) const;
  Dummy<int(Output::A)> A() const;
  Dummy<int(Output::b)> b() const;

protected:
  void update_(const taskvm::internal::UnifiedEnumValue& u);
  void fillOutputDependencies();
  void fillInternalDependencies();
  virtual void updateMatrices() = 0;

  int A_, b_;

private:
  std::string name_;
};

/** Mockup for a constraint J\dot{q} + \dot{e}*/
class KinematicLinearizedConstraint : public LinearConstraint
{
public:
  KinematicLinearizedConstraint(const std::string& name, std::shared_ptr<FunctionMockup> function);

protected:
  void fillUpdateDependencies();
  void updateMatrices() override;

private:
  std::shared_ptr<FunctionMockup> function_;
};

/** Mockup for a constraint J\ddot{q} + \dot{J}\dot{q} + \ddot{e}*/
class DynamicLinearizedConstraint : public LinearConstraint
{
public:
  DynamicLinearizedConstraint(const std::string& name, std::shared_ptr<FunctionMockup> function);

protected:
  void fillUpdateDependencies();
  void updateMatrices() override;

private:
  std::shared_ptr<FunctionMockup> function_;
};

/** Mockup for a constraint M\ddot{q} + N(q,\dot{q})*/
class DynamicEquation : public LinearConstraint
{
public:
  DynamicEquation(const std::string& name, std::shared_ptr<RobotMockup> robot);

protected:
  void fillUpdateDependencies();
  void updateMatrices() override;

private:
  std::shared_ptr<RobotMockup> robot_;
};
