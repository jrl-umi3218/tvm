#include <tvm/graph/CallGraph.h>
#include <tvm/graph/abstract/Node.h>

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
  void setValue(double) {}
  double value() const  { return 0; }
};

class RobotMockup : public tvm::graph::abstract::Node<RobotMockup>
{
public:
  SET_UPDATES(RobotMockup, Kinematics, Velocity, Dynamics, Acceleration)
  SET_OUTPUTS(RobotMockup, K1, K2, K3, V1, V2, D1, D2, A1)

  RobotMockup();

  Dummy<static_cast<int>(Output::K1)> getK1() const { return k; }
  Dummy<static_cast<int>(Output::K2)> getK2() const { return k; }
  Dummy<static_cast<int>(Output::K3)> getK3() const { return k; }
  Dummy<static_cast<int>(Output::V1)> getV1() const { return v; }
  Dummy<static_cast<int>(Output::V2)> getV2() const { return v; }
  Dummy<static_cast<int>(Output::D1)> getD1() const { return d; }
  Dummy<static_cast<int>(Output::D2)> getD2() const { return d; }
  Dummy<static_cast<int>(Output::A1)> getA1() const { return a; }

protected:
  void updateK() { std::cout << "update robot kinematics" << std::endl; ++k; }
  void updateV() { std::cout << "update robot velocity" << std::endl; ++v; }
  void updateD() { std::cout << "update robot dynamics" << std::endl; ++d; }
  void updateA() { std::cout << "update robot acceleration" << std::endl; ++a; }

  VariableMockup q;
  int k, v, d, a;
};

/** Base class for a time-dependent function*/
class FunctionMockup : public tvm::graph::abstract::Node<FunctionMockup>
{
public:
  SET_UPDATES(FunctionMockup, Value, Velocity, JDot)
  SET_OUTPUTS(FunctionMockup, Value, Jacobian, Velocity, NormalAcceleration, JDot)

  Dummy<int(Output::Value)> value() const { return val; }
  Dummy<int(Output::Value)> jacobian() const { return j; }
  Dummy<int(Output::Value)> velocity() const { return vel; }
  Dummy<int(Output::Value)> normalAcc() const { return na; }
  Dummy<int(Output::Value)> Jdot() const { return jdot; }
protected:
  FunctionMockup();

  virtual void updateValue() {}
  virtual void updateVelocity() {}
  virtual void updateJDot() {}

  int val, j, vel, na, jdot;

};

/** Base class for a function relying on a robot to compute its own outputs*/
class RobotFunction : public FunctionMockup
{
protected:
  RobotFunction(std::shared_ptr<RobotMockup> robot);

protected:
  std::shared_ptr<RobotMockup> robot_;
};

class SomeRobotFunction1 : public RobotFunction
{
public:
  SomeRobotFunction1(std::shared_ptr<RobotMockup> robot);
protected:
  void updateValue() override;
  void updateVelocity() override;
  void updateJDot() override;
};

class SomeRobotFunction2 : public RobotFunction
{
public:
  SomeRobotFunction2(std::shared_ptr<RobotMockup> robot);
protected:
  void updateValue() override;
  void updateVelocity() override;
};

class BadRobotFunction : public RobotFunction
{
public:
  BadRobotFunction(std::shared_ptr<RobotMockup> robot);
protected:
  void update();
};

/** Base class for a linear expression Ax+b*/
class LinearConstraint: public tvm::graph::abstract::Node<LinearConstraint>
{
public:
  SET_OUTPUTS(LinearConstraint, Value, A, b)
  SET_UPDATES(LinearConstraint, Matrices)

  LinearConstraint(const std::string& name);

  //example of method with argument
  Dummy<int(Output::Value)> value(int x) const;
  Dummy<int(Output::A)> A() const;
  Dummy<int(Output::b)> b() const;

protected:
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
  void updateMatrices() override;

private:
  std::shared_ptr<RobotMockup> robot_;
};


/** Base class for a linear expression Ax+b*/
class BetterLinearConstraint : public tvm::graph::abstract::Node<BetterLinearConstraint>
{
public:
  SET_OUTPUTS(BetterLinearConstraint, Value, A, b)

  BetterLinearConstraint(const std::string& name);

  //example of method with argument
  Dummy<int(Output::Value)> value(int x) const;
  virtual Dummy<int(Output::A)> A() const;
  virtual Dummy<int(Output::b)> b() const;

protected:
  int A_, b_;

private:
  std::string name_;
};

/** Mockup for a constraint J\dot{q} + \dot{e}*/
class BetterKinematicLinearizedConstraint : public BetterLinearConstraint
{
public:
  BetterKinematicLinearizedConstraint(const std::string& name, std::shared_ptr<FunctionMockup> function);

  virtual Dummy<int(Output::A)> A() const override;
  virtual Dummy<int(Output::b)> b() const override;

private:
  std::shared_ptr<FunctionMockup> function_;
};

/** Mockup for a constraint J\ddot{q} + \dot{J}\dot{q} + \ddot{e}*/
class BetterDynamicLinearizedConstraint : public BetterLinearConstraint
{
public:
  SET_UPDATES(BetterDynamicLinearizedConstraint, Updateb)

  BetterDynamicLinearizedConstraint(const std::string& name, std::shared_ptr<FunctionMockup> function);
  void updateb();

  virtual Dummy<int(Output::A)> A() const override;

protected:

private:
  std::shared_ptr<FunctionMockup> function_;
};
