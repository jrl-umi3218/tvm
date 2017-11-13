#include <tvm/function/BasicLinearFunction.h>

#include <tvm/Variable.h>

namespace tvm
{

namespace function
{

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
  this->b(b);
}

void BasicLinearFunction::A(const MatrixConstRef& A, const Variable& x)
{
  if (A.rows() == size() && A.cols() == x.size())
    jacobian_.at(&x) = A;
  else
    throw std::runtime_error("Matrix A doesn't have the good size.");
}

void BasicLinearFunction::A(const MatrixConstRef& A)
{
  if (variables().size() == 1)
    this->A(A, *variables()[0].get());
  else
    throw std::runtime_error("You can use this method only for constraints with one variable.");
}

void BasicLinearFunction::b(const VectorConstRef& b)
{
  if (b.size() == size())
    this->b_ = b;
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
  jacobian_.at(x.get()).properties({ tvm::internal::MatrixProperties::Constness(true) });
}

}  // namespace function

}  // namespace tvm
