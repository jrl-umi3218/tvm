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
    Variable& x_;
    Variable& y_;
    Variable& dx_;
    Variable& dy_;
  };

  DotProduct::DotProduct(VariablePtr x, VariablePtr y)
    : Function(1)
    , x_(*x)
    , y_(*y)
    , dx_(*dot(x))
    , dy_(*dot(y))
  {
    registerUpdates(Update::Value, &DotProduct::updateValue,
                    Update::Jacobian, &DotProduct::updateJacobian,
                    Update::VelocityAndNormalAcc, &DotProduct::updateVelocityAndNormalAcc,
                    Update::JDot, &DotProduct::updateJDot);
    addOutputDependency<DotProduct>(Output::Value, Update::Value);
    addOutputDependency<DotProduct>(Output::Jacobian, Update::Jacobian);
    addOutputDependency<DotProduct>({ Output::Velocity, Output::NormalAcceleration }, Update::VelocityAndNormalAcc);
    addOutputDependency<DotProduct>(Output::JDot, Update::JDot);
    addVariable(x, true);
    addVariable(y, true);
  }

  void DotProduct::updateValue()
  {
    value_ = x_.value().transpose() * y_.value();
  }

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
}


#include<iostream>
using namespace tvm;
using namespace tvm::example;

int main()
{
  Space R3(3);
  VariablePtr x = R3.createVariable("x"); x << 1, 2, 3;
  VariablePtr y = R3.createVariable("y"); y << 1, 2, 3;
  DotProduct dp(x, y);
  dp.updateValue();
  dp.updateJacobian();
  std::cout << dp.value() << std::endl;
  std::cout << dp.jacobian(*x) << std::endl;
  std::cout << dp.jacobian(*y) << std::endl;
}