#include "LinearConstraint.h"
#include "Variable.h"

namespace tvm
{
  LinearConstraint::LinearConstraint(ConstraintType ct, ConstraintRHS cr, int m)
    :Constraint(ct,cr,m)
  {
    registerUpdates(LinearConstraint::Update::Value,&LinearConstraint::updateValue);
    addOutputDependency<LinearConstraint>(FirstOrderProvider::Output::Value, LinearConstraint::Update::Value);
  }

  void LinearConstraint::updateValue()
  {
    value_.setZero();
    for (auto v : variables())
      value_ += jacobian(*v) * v->value();
  }



  BasicLinearConstraint::BasicLinearConstraint(const MatrixConstRef& A, VariablePtr x, ConstraintType ct)
    : BasicLinearConstraint({ A }, { x }, ct)
  {
  }

  BasicLinearConstraint::BasicLinearConstraint(std::initializer_list<MatrixConstRef> A, std::initializer_list<VariablePtr> x, ConstraintType ct)
    : LinearConstraint(ct, ConstraintRHS::ZERO, static_cast<int>(A.begin()->rows()))
  {
    if (ct == ConstraintType::DOUBLE_SIDED)
      throw std::runtime_error("This constructor is only for single-sided constraints.");
    if (A.size() != x.size())
      throw std::runtime_error("The number of matrices and variables is incoherent.");
    auto v = x.begin();
    for (const Eigen::MatrixXd& a : A)
    {
      add(a, *v);
      ++v;
    }
  }

  BasicLinearConstraint::BasicLinearConstraint(const MatrixConstRef& A, VariablePtr x, const VectorConstRef& b, ConstraintType ct, ConstraintRHS cr)
    : BasicLinearConstraint({ A }, { x }, b, ct, cr)
  {
  }

  BasicLinearConstraint::BasicLinearConstraint(std::initializer_list<MatrixConstRef> A, std::initializer_list<VariablePtr> x, const VectorConstRef& b, ConstraintType ct, ConstraintRHS cr)
    : LinearConstraint(ct, cr, static_cast<int>(A.begin()->rows()))
  {
    if (ct == ConstraintType::DOUBLE_SIDED)
      throw std::runtime_error("This constructor is only for single-sided constraints.");
    if (cr == ConstraintRHS::ZERO)
      throw std::runtime_error("ConstraintRHS::ZERO is not a valid input for this constructor. Please use the constructor for Ax=0, Ax<=0 and Ax>=0 instead.");
    if (A.size() != x.size())
      throw std::runtime_error("The number of matrices and variables is incoherent.");
    if (b.size() != size())
      throw std::runtime_error("Vector b doesn't have the good size.");

    auto v = x.begin();
    for (const Eigen::MatrixXd& a : A)
    {
      add(a, *v);
      ++v;
    }
    setb(b);
  }

  BasicLinearConstraint::BasicLinearConstraint(const MatrixConstRef& A, VariablePtr x, 
                                               const VectorConstRef& l, const VectorConstRef& u, ConstraintRHS cr)
    :BasicLinearConstraint({ A }, { x }, l, u, cr)
  {
  }

  BasicLinearConstraint::BasicLinearConstraint(std::initializer_list<MatrixConstRef> A, std::initializer_list<VariablePtr> x, 
                                               const VectorConstRef& l, const VectorConstRef& u, ConstraintRHS cr)
    : LinearConstraint(ConstraintType::DOUBLE_SIDED, cr, static_cast<int>(A.begin()->rows()))
  {
    if (cr == ConstraintRHS::ZERO)
      throw std::runtime_error("ConstraintRHS::ZERO is not a valid input for this constructor. Please use the constructor for Ax=0, Ax<=0 and Ax>=0 instead.");
    if (A.size() != x.size())
      throw std::runtime_error("The number of matrices and variables is incoherent.");
    if (l.size() != size())
      throw std::runtime_error("Vector l doesn't have the good size.");
    if (u.size() != size())
      throw std::runtime_error("Vector u doesn't have the good size.");

    auto v = x.begin();
    for (const Eigen::MatrixXd& a : A)
    {
      add(a, *v);
      ++v;
    }

    setl(l);
    setu(u);
  }

  void BasicLinearConstraint::setA(const MatrixConstRef& A, const Variable& x)
  {
    if (A.rows() == size() && A.cols() == x.size())
      jacobian_.at(&x) = A;
    else
      throw std::runtime_error("Matrix A doesn't have the good size.");
  }

  void BasicLinearConstraint::setA(const MatrixConstRef& A)
  {
    if (variables().size() == 1)
      setA(A, *variables()[0].get());
    else
      throw std::runtime_error("You can use this method only for constraints with one variable.");
  }

  void BasicLinearConstraint::setb(const VectorConstRef& b)
  {
    if (constraintType() != ConstraintType::DOUBLE_SIDED && constraintRhs() != ConstraintRHS::ZERO)
    {
      if (b.size() == size())
      {
        switch (constraintType())
        {
        case ConstraintType::EQUAL: e_ = b; break;
        case ConstraintType::GREATER_THAN: l_ = b; break;
        case ConstraintType::LOWER_THAN: u_ = b; break;
        }
      }
      else
        throw std::runtime_error("Vector b doesn't have the correct size.");
    }
    else
      throw std::runtime_error("setb is not allowed for this constraint.");
  }

  void BasicLinearConstraint::setl(const VectorConstRef& l)
  {
    if (constraintType() == ConstraintType::DOUBLE_SIDED && constraintRhs() != ConstraintRHS::ZERO)
    {
      if (l.size() == size())
        l_ = l;
      else
        throw std::runtime_error("Vector l doesn't have the correct size.");
    }
    else
      throw std::runtime_error("setl is not allowed for this constraint.");
  }

  void BasicLinearConstraint::setu(const VectorConstRef& u)
  {
    if (constraintType() == ConstraintType::DOUBLE_SIDED && constraintRhs() != ConstraintRHS::ZERO)
    {
      if (u.size() == size())
        u_ = u;
      else
        throw std::runtime_error("Vector u doesn't have the correct size.");
    }
    else
      throw std::runtime_error("setu is not allowed for this constraint.");
  }

  void BasicLinearConstraint::add(const Eigen::MatrixXd& A, VariablePtr x)
  {
    if (!x->space().isEuclidean())
      throw std::runtime_error("We allow linear constraint only on Euclidean variables.");
    if (A.rows() != size())
      throw std::runtime_error("Matrix A doesn't have coherent row size.");
    if (A.cols() != x->size())
      throw std::runtime_error("Matrix A doesn't have its column size coherent with its corresponding variable.");
    addVariable(x, true);
    jacobian_.at(x.get()) = A;
    jacobian_.at(x.get()).properties({ MatrixProperties::Constness(true) });
  }
}
