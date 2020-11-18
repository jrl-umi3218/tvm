/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/function/BasicLinearFunction.h>

#include <tvm/Variable.h>

namespace tvm
{

namespace function
{

BasicLinearFunction::BasicLinearFunction(const MatrixConstRef & A, VariablePtr x)
: BasicLinearFunction({A}, std::vector<VariablePtr>{x})
{}

BasicLinearFunction::BasicLinearFunction(const std::vector<MatrixConstRef> & A, const std::vector<VariablePtr> & x)
: BasicLinearFunction(A, x, Eigen::VectorXd::Zero(A.begin()->rows()))
{
  this->b_.properties({tvm::internal::MatrixProperties::Constness(true), tvm::internal::MatrixProperties::ZERO});
}

BasicLinearFunction::BasicLinearFunction(const MatrixConstRef & A, VariablePtr x, const VectorConstRef & b)
: BasicLinearFunction({A}, std::vector<VariablePtr>{x}, b)
{}

BasicLinearFunction::BasicLinearFunction(const std::vector<MatrixConstRef> & A,
                                         const std::vector<VariablePtr> & x,
                                         const VectorConstRef & b)
: LinearFunction(static_cast<int>(A.begin()->rows()))
{
  if(A.size() != x.size())
    throw std::runtime_error("The number of matrices and variables is incoherent.");

  auto v = x.begin();
  for(const auto & a : A)
  {
    add(a, *v);
    ++v;
  }
  this->b(b);
  setDerivativesToZero();
}

BasicLinearFunction::BasicLinearFunction(int m, VariablePtr x) : BasicLinearFunction(m, std::vector<VariablePtr>{x}) {}

BasicLinearFunction::BasicLinearFunction(int m, const std::vector<VariablePtr> & x) : LinearFunction(m)
{
  for(auto & v : x)
  {
    addVariable(v, true);
    jacobian_.at(v.get()).properties({tvm::internal::MatrixProperties::Constness(true)});
  }
  setDerivativesToZero();
}

void BasicLinearFunction::A(const MatrixConstRef & A, const Variable & x, const tvm::internal::MatrixProperties & p)
{
  if(A.rows() == size() && A.cols() == x.size())
  {
    jacobian_.at(&x) = A;
    jacobian_.at(&x).properties(p);
  }
  else
    throw std::runtime_error("Matrix A doesn't have the good size.");
}

void BasicLinearFunction::A(const MatrixConstRef & A, const tvm::internal::MatrixProperties & p)
{
  if(variables().numberOfVariables() == 1)
    this->A(A, *variables()[0].get(), p);
  else
    throw std::runtime_error("You can use this method only for constraints with one variable.");
}

void BasicLinearFunction::b(const VectorConstRef & b, const tvm::internal::MatrixProperties & p)
{
  if(b.size() == size())
  {
    this->b_ = b;
    this->b_.properties(p);
  }
  else
    throw std::runtime_error("Vector b doesn't have the correct size.");
}

} // namespace function

} // namespace tvm
