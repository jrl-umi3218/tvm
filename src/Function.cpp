#include "errors.h"
#include "Function.h"
#include "Variable.h"

namespace taskvm
{
  const Eigen::VectorXd& Function::velocity() const
  {
    if (isOutputEnabled((int)Output::Velocity))
      return velocityNoCheck();
    else
      throw UnusedOutput(/*description*/); //TODO add description of the error
  }

  const Eigen::VectorXd& Function::normalAcceleration() const
  {
    if (isOutputEnabled((int)Output::NormalAcceleration))
      return normalAccelerationNoCheck();
    else
      throw UnusedOutput(/*description*/); //TODO add description of the error
  }

  const Eigen::MatrixXd& Function::JDot(const Variable& x) const
  {
    if (isOutputEnabled((int)Output::JDot))
      return JDotNoCheck(x);
    else
      throw UnusedOutput(/*description*/); //TODO add description of the error
  }

  const Eigen::VectorXd& Function::velocityNoCheck() const
  {
    return velocity_;
  }

  const Eigen::VectorXd& Function::normalAccelerationNoCheck() const
  {
    return normalAcceleration_;
  }

  const Eigen::MatrixXd& Function::JDotNoCheck(const Variable& x) const
  {
    return JDot_.at(&x);
  }


  void Function::resizeCache()
  {
    FirstOrderProvider::resizeCache();

    if (isOutputEnabled((int)Output::Velocity))
      velocity_.resize(size());

    if (isOutputEnabled((int)Output::NormalAcceleration))
      normalAcceleration_.resize(size());

    if (isOutputEnabled((int)Output::JDot))
    {
      for (auto v : variables())
        JDot_[v.get()].resize(size(), v->size());
    }
  }

  void Function::addVariable_(std::shared_ptr<Variable> v)
  {
    JDot_[v.get()].resize(size(), v->size());
  }

  void Function::removeVariable_(std::shared_ptr<Variable> v)
  {
    JDot_.erase(v.get());
  }
}
