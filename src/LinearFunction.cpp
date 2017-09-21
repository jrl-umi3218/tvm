#include "LinearFunction.h"
#include "Variable.h"

namespace tvm
{
  LinearFunction::LinearFunction(int m)
    :Function(m)
  {
    registerUpdates(LinearFunction::Update::Value, &LinearFunction::updateValue);
    registerUpdates(LinearFunction::Update::Velocity, &LinearFunction::updateVelocity);
    addOutputDependency<LinearFunction>(FirstOrderProvider::Output::Value, LinearFunction::Update::Value);
    addOutputDependency<LinearFunction>(Function::Output::Velocity, LinearFunction::Update::Velocity);
    setDerivativesToZero();
  }

  void LinearFunction::updateValue()
  {
    updateValue_();
  }

  void LinearFunction::updateVelocity()
  {
    updateVelocity_();
  }

  void LinearFunction::resizeCache()
  {
    Function::resizeCache();
    b_.resize(size());
    setDerivativesToZero();
  }

  void LinearFunction::updateValue_()
  {
    value_ = b_;
    for (auto v : variables())
      value_ += jacobian(*v) * v->value();
  }

  void LinearFunction::updateVelocity_()
  {
    value_.setZero();
    for (auto v : variables())
      value_ += jacobian(*v) * dot(v)->value();
  }

  void LinearFunction::setDerivativesToZero()
  {
    normalAcceleration_.setZero();
    for (const auto& v : variables())
      JDot_[v.get()].setZero();
  }

  BasicLinearFunction::BasicLinearFunction(const MatrixConstRef& A, VariablePtr x)
    : BasicLinearFunction({A}, {x})
  {
  }

  BasicLinearFunction::BasicLinearFunction(std::initializer_list<MatrixConstRef> A, std::initializer_list<VariablePtr> x)
    : BasicLinearFunction(A, x, Eigen::VectorXd::Zero(A.begin()->rows()))
  {
  }

  BasicLinearFunction::BasicLinearFunction(const MatrixConstRef& A, VariablePtr x, const VectorConstRef& b)
    : BasicLinearFunction({A}, {x}, b)
  {
  }

  BasicLinearFunction::BasicLinearFunction(std::initializer_list<MatrixConstRef> A, std::initializer_list<VariablePtr> x, const VectorConstRef& b)
    : LinearFunction(static_cast<int>(A.begin()->rows()))
  {
    if (A.size() != x.size())
      throw std::runtime_error("The number of matrices and variables is incoherent.");

    auto v = x.begin();
    for (const Eigen::MatrixXd& a : A)
    {
      add(a, *v);
      ++v;
    }
  }

  void BasicLinearFunction::setA(const MatrixConstRef& A, const Variable& x)
  {
    if (A.rows() == size() && A.cols() == x.size())
      jacobian_.at(&x) = A;
    else
      throw std::runtime_error("Matrix A doesn't have the good size.");
  }

  void BasicLinearFunction::setA(const MatrixConstRef& A)
  {
    if (variables().size() == 1)
      setA(A, *variables()[0].get());
    else
      throw std::runtime_error("You can use this method only for constraints with one variable.");
  }

  void BasicLinearFunction::setb(const VectorConstRef& b)
  {
    if (b.size() == size())
      b_ = b;
    else
      throw std::runtime_error("Vector b doesn't have the correct size.");
  }

  void BasicLinearFunction::add(const Eigen::MatrixXd& A, VariablePtr x)
  {
    if (!x->space().isEuclidean())
      throw std::runtime_error("We allow linear function only on Euclidean variables.");
    if (A.rows() != size())
      throw std::runtime_error("Matrix A doesn't have coherent row size.");
    if (A.cols() != x->size())
      throw std::runtime_error("Matrix A doesn't have its column size coherent with its corresponding variable.");
    addVariable(x, true);
    jacobian_.at(x.get()) = A;
    jacobian_.at(x.get()).properties({ MatrixProperties::Constness(true) });
  }


  IdentityFunction::IdentityFunction(VariablePtr x)
    : BasicLinearFunction(Eigen::MatrixXd::Identity(x->size(), x->size()), x)
  {
    jacobian_.begin()->second.properties({ MatrixProperties::Shape::IDENTITY });
  }

  void IdentityFunction::setA(const MatrixConstRef & A, const Variable & x)
  {
    throw std::runtime_error("You can not change the A matrix on a identity function");
  }

  void IdentityFunction::setA(const MatrixConstRef & A)
  {
    throw std::runtime_error("You can not change the A matrix on a identity function");
  }

  void IdentityFunction::setb(const VectorConstRef& b)
  {
    throw std::runtime_error("You can not change the b vector on a identity function");
  }

  void IdentityFunction::updateValue_()
  {
    value_ = variables()[0]->value();
  }

  void IdentityFunction::updateVelocity_()
  {
    velocity_ = dot(variables()[0])->value();
  }
}