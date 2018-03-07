#include <tvm/constraint/BasicLinearConstraint.h>

#include <tvm/Variable.h>

namespace tvm
{

namespace constraint
{

  BasicLinearConstraint::BasicLinearConstraint(const MatrixConstRef& A, VariablePtr x, Type ct)
    : BasicLinearConstraint(std::vector<MatrixConstRef>{ A }, { x }, ct)
  {
  }

  BasicLinearConstraint::BasicLinearConstraint(const std::vector<MatrixConstRef>& A, const std::vector<VariablePtr>& x, Type ct)
    : LinearConstraint(ct, RHS::ZERO, static_cast<int>(A.begin()->rows()))
  {
    if (ct == Type::DOUBLE_SIDED)
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

  BasicLinearConstraint::BasicLinearConstraint(const MatrixConstRef& A, VariablePtr x, const VectorConstRef& b, Type ct, RHS cr)
    : BasicLinearConstraint(std::vector<MatrixConstRef>{ A }, { x }, b, ct, cr)
  {
  }

  BasicLinearConstraint::BasicLinearConstraint(const std::vector<MatrixConstRef>& A, const std::vector<VariablePtr>& x, const VectorConstRef& b, Type ct, RHS cr)
    : LinearConstraint(ct, cr, static_cast<int>(A.begin()->rows()))
  {
    if (ct == Type::DOUBLE_SIDED)
      throw std::runtime_error("This constructor is only for single-sided constraints.");
    if (cr == RHS::ZERO)
      throw std::runtime_error("RHS::ZERO is not a valid input for this constructor. Please use the constructor for Ax=0, Ax<=0 and Ax>=0 instead.");
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
    this->b(b);
  }

  BasicLinearConstraint::BasicLinearConstraint(const MatrixConstRef& A, VariablePtr x,
                                               const VectorConstRef& l, const VectorConstRef& u, RHS cr)
    :BasicLinearConstraint(std::vector<MatrixConstRef>{ A }, { x }, l, u, cr)
  {
  }

  BasicLinearConstraint::BasicLinearConstraint(const std::vector<MatrixConstRef>& A, const std::vector<VariablePtr>& x,
                                               const VectorConstRef& l, const VectorConstRef& u, RHS cr)
    : LinearConstraint(Type::DOUBLE_SIDED, cr, static_cast<int>(A.begin()->rows()))
  {
    if (cr == RHS::ZERO)
      throw std::runtime_error("RHS::ZERO is not a valid input for this constructor. Please use the constructor for Ax=0, Ax<=0 and Ax>=0 instead.");
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

    this->l(l);
    this->u(u);
  }

  BasicLinearConstraint::BasicLinearConstraint(int m, std::vector<VariablePtr>& x, Type ct, RHS cr)
    : LinearConstraint(ct, cr, m)
  {
    for (const auto& v : x)
    {
      addVariable(v, true);
    }
  }

  void BasicLinearConstraint::A(const MatrixConstRef& A, const Variable& x)
  {
    if (A.rows() == size() && A.cols() == x.size())
      jacobian_.at(&x) = A;
    else
      throw std::runtime_error("Matrix A doesn't have the good size.");
  }

  void BasicLinearConstraint::A(const MatrixConstRef& A)
  {
    if (variables().size() == 1)
    {
      this->A(A, *variables()[0].get());
    }
    else
      throw std::runtime_error("You can use this method only for constraints with one variable.");
  }

  void BasicLinearConstraint::b(const VectorConstRef& b)
  {
    if (type() != Type::DOUBLE_SIDED && rhs() != RHS::ZERO)
    {
      if (b.size() == size())
      {
        switch (type())
        {
        case Type::EQUAL: eRef() = b; break;
        case Type::GREATER_THAN: lRef() = b; break;
        case Type::LOWER_THAN: uRef() = b; break;
        default: break;
        }
      }
      else
        throw std::runtime_error("Vector b doesn't have the correct size.");
    }
    else
      throw std::runtime_error("setb is not allowed for this constraint.");
  }

  void BasicLinearConstraint::l(const VectorConstRef& l)
  {
    if (type() == Type::DOUBLE_SIDED && rhs() != RHS::ZERO)
    {
      if (l.size() == size())
      {
        lRef() = l;
      }
      else
      {
        throw std::runtime_error("Vector l doesn't have the correct size.");
      }
    }
    else
    {
      throw std::runtime_error("setl is not allowed for this constraint.");
    }
  }

  void BasicLinearConstraint::u(const VectorConstRef& u)
  {
    if (type() == Type::DOUBLE_SIDED && rhs() != RHS::ZERO)
    {
      if (u.size() == size())
      {
        uRef() = u;
      }
      else
      {
        throw std::runtime_error("Vector u doesn't have the correct size.");
      }
    }
    else
      throw std::runtime_error("setu is not allowed for this constraint.");
  }

  void BasicLinearConstraint::add(const Eigen::MatrixXd& A, VariablePtr x)
  {
    if (!x->space().isEuclidean() && x->isBasePrimitive())
      throw std::runtime_error("We allow linear constraint only on Euclidean variables.");
    if (A.rows() != size())
      throw std::runtime_error("Matrix A doesn't have coherent row size.");
    if (A.cols() != x->size())
      throw std::runtime_error("Matrix A doesn't have its column size coherent with its corresponding variable.");
    addVariable(x, true);
    jacobian_.at(x.get()) = A;
    jacobian_.at(x.get()).properties({ tvm::internal::MatrixProperties::Constness(true) });
  }

}  // namespace constraint

}  // namespace tvm
