#include <functional>
#include <iostream>
#include <memory>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <tvm/Variable.h>
#include <tvm/VariableVector.h>
#include <tvm/constraint/BasicLinearConstraint.h>
#include <tvm/scheme/internal/Assignment.h>
#include <tvm/scheme/internal/AssignmentTarget.h>

#include <Eigen/Core>
#include <Eigen/QR>

//FIXME see src/Assignment.cpp
static const double large = 1e6;

struct Constraints
{
  Eigen::VectorXd p0;
  Eigen::VectorXd pl;
  Eigen::VectorXd pu;

  std::shared_ptr<tvm::constraint::BasicLinearConstraint> Ax_eq_0;
  std::shared_ptr<tvm::constraint::BasicLinearConstraint> Ax_geq_0;
  std::shared_ptr<tvm::constraint::BasicLinearConstraint> Ax_leq_0;
  std::shared_ptr<tvm::constraint::BasicLinearConstraint> Ax_eq_b;
  std::shared_ptr<tvm::constraint::BasicLinearConstraint> Ax_geq_b;
  std::shared_ptr<tvm::constraint::BasicLinearConstraint> Ax_leq_b;
  std::shared_ptr<tvm::constraint::BasicLinearConstraint> Ax_eq_minus_b;
  std::shared_ptr<tvm::constraint::BasicLinearConstraint> Ax_geq_minus_b;
  std::shared_ptr<tvm::constraint::BasicLinearConstraint> Ax_leq_minus_b;
  std::shared_ptr<tvm::constraint::BasicLinearConstraint> l_leq_Ax_leq_u;
  std::shared_ptr<tvm::constraint::BasicLinearConstraint> minus_l_leq_Ax_leq_minus_u;
};

struct Memory
{
  Memory(int m, int n) : A(m, n), b(m), l(m), u(m)
  {
    A.setZero();
    b.setZero();
    l.setZero();
    u.setZero();
  }

  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::VectorXd l;
  Eigen::VectorXd u;
};

//Check if the constraint is satisfied for the current value of the variable
bool check(std::shared_ptr<tvm::constraint::BasicLinearConstraint> c, const Eigen::VectorXd& x)
{
  const double eps = 1e-12;
  c->variables()[0]->value(x);
  c->updateValue();
  auto v = c->value();
  if (c->type() == tvm::constraint::Type::DOUBLE_SIDED)
  {
    if (c->rhs() == tvm::constraint::RHS::AS_GIVEN)
      return (c->l().array()-eps <= v.array()).all() && (v.array() <= c->u().array()+eps).all();
    else
      return (-c->l().array()-eps <= v.array()).all() && (v.array() <= -c->u().array()+eps).all();
  }
  else
  {
    std::function<bool(const Eigen::VectorXd&, const Eigen::VectorXd&)> comp;
    const Eigen::VectorXd&(tvm::constraint::BasicLinearConstraint::*rhs)() const;
    switch (c->type())
    {
    case tvm::constraint::Type::EQUAL: comp = [](const Eigen::VectorXd& u, const Eigen::VectorXd& v) {return u.isApprox(v); }; rhs =& tvm::constraint::BasicLinearConstraint::e;  break;
    case tvm::constraint::Type::GREATER_THAN: comp = [eps](const Eigen::VectorXd& u, const Eigen::VectorXd& v) {return (u.array() + eps >= v.array()).all(); }; rhs = &tvm::constraint::BasicLinearConstraint::l; break;
    case tvm::constraint::Type::LOWER_THAN: comp = [eps](const Eigen::VectorXd& u, const Eigen::VectorXd& v) {return (u.array() - eps <= v.array()).all(); }; rhs = &tvm::constraint::BasicLinearConstraint::u; break;
    default: break;
    }
    switch (c->rhs())
    {
    case tvm::constraint::RHS::AS_GIVEN: return comp(v, (c.get()->*rhs)()); break;
    case tvm::constraint::RHS::OPPOSITE: return comp(v, -(c.get()->*rhs)()); break;
    case tvm::constraint::RHS::ZERO: return comp(v, Eigen::VectorXd::Zero(c->size())); break;
    default:
      return false;
    }
  }
}

bool check(const Memory& mem, tvm::constraint::Type ct, tvm::constraint::RHS cr, const Eigen::VectorXd& x)
{
  const double eps = 1e-12;
  Eigen::VectorXd v = mem.A*x;
  if (ct == tvm::constraint::Type::DOUBLE_SIDED)
  {
    if (cr == tvm::constraint::RHS::AS_GIVEN)
      return (mem.l.array() - eps <= v.array()).all() && (v.array() <= mem.u.array() + eps).all();
    else
      return (-mem.l.array() - eps <= v.array()).all() && (v.array() <= -mem.u.array() + eps).all();
  }
  else
  {
    std::function<bool(const Eigen::VectorXd&, const Eigen::VectorXd&)> comp;
    switch (ct)
    {
    case tvm::constraint::Type::EQUAL: comp = [](const Eigen::VectorXd& u, const Eigen::VectorXd& v) {return u.isApprox(v); };  break;
    case tvm::constraint::Type::GREATER_THAN: comp = [eps](const Eigen::VectorXd& u, const Eigen::VectorXd& v) {return (u.array() + eps >= v.array()).all(); }; break;
    case tvm::constraint::Type::LOWER_THAN: comp = [eps](const Eigen::VectorXd& u, const Eigen::VectorXd& v) {return (u.array() - eps <= v.array()).all(); }; break;
    default: break;
    }
    switch (cr)
    {
    case tvm::constraint::RHS::AS_GIVEN: return comp(v, mem.b); break;
    case tvm::constraint::RHS::OPPOSITE: return comp(v, -mem.b); break;
    case tvm::constraint::RHS::ZERO: return comp(v, Eigen::VectorXd::Zero(mem.b.rows())); break;
    default:
      return false;
    }
  }
}

Constraints buildConstraints(int m, int n)
{
  Constraints cstr;
  tvm::VariablePtr x = tvm::Space(n).createVariable("x");

  //generate matrix
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
  Eigen::VectorXd l = -Eigen::VectorXd::Random(m).cwiseAbs();
  Eigen::VectorXd u = Eigen::VectorXd::Random(m).cwiseAbs();

  //Point p0 such that Ap0 = 0
  cstr.p0 = A.householderQr().solve(Eigen::VectorXd::Zero(m));
  //Point pl such that Apl = l
  cstr.pl = A.householderQr().solve(l);
  //Point pu such that Apu = u
  cstr.pu = A.householderQr().solve(u);

  cstr.Ax_eq_0 = std::make_shared<tvm::constraint::BasicLinearConstraint>(A, x, tvm::constraint::Type::EQUAL);
  cstr.Ax_geq_0 = std::make_shared<tvm::constraint::BasicLinearConstraint>(A, x, tvm::constraint::Type::GREATER_THAN);
  cstr.Ax_leq_0 = std::make_shared<tvm::constraint::BasicLinearConstraint>(A, x, tvm::constraint::Type::LOWER_THAN);

  cstr.Ax_eq_b = std::make_shared<tvm::constraint::BasicLinearConstraint>(A, x, l, tvm::constraint::Type::EQUAL);
  cstr.Ax_geq_b = std::make_shared<tvm::constraint::BasicLinearConstraint>(A, x, l, tvm::constraint::Type::GREATER_THAN);
  cstr.Ax_leq_b = std::make_shared<tvm::constraint::BasicLinearConstraint>(A, x, u, tvm::constraint::Type::LOWER_THAN);

  cstr.Ax_eq_minus_b = std::make_shared<tvm::constraint::BasicLinearConstraint>(A, x, -l, tvm::constraint::Type::EQUAL, tvm::constraint::RHS::OPPOSITE);
  cstr.Ax_geq_minus_b = std::make_shared<tvm::constraint::BasicLinearConstraint>(A, x, -l, tvm::constraint::Type::GREATER_THAN, tvm::constraint::RHS::OPPOSITE);
  cstr.Ax_leq_minus_b = std::make_shared<tvm::constraint::BasicLinearConstraint>(A, x, -u, tvm::constraint::Type::LOWER_THAN, tvm::constraint::RHS::OPPOSITE);

  cstr.l_leq_Ax_leq_u = std::make_shared<tvm::constraint::BasicLinearConstraint>(A, x, l, u);
  cstr.minus_l_leq_Ax_leq_minus_u = std::make_shared<tvm::constraint::BasicLinearConstraint>(A, x, -l, -u, tvm::constraint::RHS::OPPOSITE);
  return cstr;
}

TEST_CASE("Test assigments")
{
  Constraints cstr = buildConstraints(3, 7);

  {
    auto mem = std::make_shared<Memory>(6, 7);
    //assignment to a target with convention l <= Ax <= u, from convention Ax >= -b
    auto range = std::make_shared<tvm::Range>(2, 3);
    tvm::scheme::internal::AssignmentTarget at(range, { mem, &mem->A }, { mem, &mem->l }, { mem, &mem->u }, tvm::constraint::RHS::AS_GIVEN);
    auto req = std::make_shared<tvm::requirements::SolvingRequirements>(tvm::requirements::Weight(2.));
    tvm::VariableVector vv(cstr.Ax_eq_0->variables());
    tvm::scheme::internal::Assignment a(cstr.Ax_geq_minus_b, req, at, vv);
    a.run();

    {
      const auto & cstr_A = cstr.Ax_geq_minus_b->jacobian(*cstr.Ax_geq_minus_b->variables()[0]);
      const auto & cstr_l = cstr.Ax_geq_minus_b->l();
      FAST_CHECK_EQ(mem->A.block(range->start, 0, 3, 7), sqrt(2)*cstr_A);
      FAST_CHECK_EQ(mem->l.block(range->start, 0, 3, 1), -sqrt(2)*cstr_l);
      FAST_CHECK_EQ(mem->u.block(range->start, 0, 3, 1), sqrt(2)*Eigen::VectorXd(3).setConstant(large));
    }

    FAST_CHECK_EQ(check(cstr.Ax_geq_minus_b, cstr.p0), check(*mem.get(), at.constraintType(), at.constraintRhs(), cstr.p0));
    FAST_CHECK_EQ(check(cstr.Ax_geq_minus_b, cstr.pl), check(*mem.get(), at.constraintType(), at.constraintRhs(), cstr.pl));
    FAST_CHECK_EQ(check(cstr.Ax_geq_minus_b, cstr.pu), check(*mem.get(), at.constraintType(), at.constraintRhs(), cstr.pu));

    //now we change the range of the target and refresh the assignment
    range->start = 0;
    a.onUpdatedTarget();
    mem->A.setZero();
    mem->l.setZero();
    mem->u.setZero();
    a.run();

    {
      const auto & cstr_A = cstr.Ax_geq_minus_b->jacobian(*cstr.Ax_geq_minus_b->variables()[0]);
      const auto & cstr_l = cstr.Ax_geq_minus_b->l();
      FAST_CHECK_EQ(mem->A.block(range->start, 0, 3, 7), sqrt(2)*cstr_A);
      FAST_CHECK_EQ(mem->l.block(range->start, 0, 3, 1), -sqrt(2)*cstr_l);
      FAST_CHECK_EQ(mem->u.block(range->start, 0, 3, 1), sqrt(2)*Eigen::VectorXd(3).setConstant(large));
    }
  }

  {
    auto mem = std::make_shared<Memory>(6, 7);
    //assignment to a target with convention Ax <= b, from convention l <= Ax <= u
    auto range = std::make_shared<tvm::Range>(0, 6); //we need double range
    tvm::scheme::internal::AssignmentTarget at(range, { mem, &mem->A }, { mem, &mem->b }, tvm::constraint::Type::LOWER_THAN, tvm::constraint::RHS::AS_GIVEN);
    Eigen::Vector3d aW = {1., 2., 3.};
    auto req = std::make_shared<tvm::requirements::SolvingRequirements>(tvm::requirements::AnisotropicWeight{ aW });
    tvm::VariableVector vv(cstr.Ax_eq_0->variables());
    tvm::scheme::internal::Assignment a(cstr.l_leq_Ax_leq_u, req, at, vv);
    a.run();

    {
      const auto & cstr_A = cstr.l_leq_Ax_leq_u->jacobian(*cstr.l_leq_Ax_leq_u->variables()[0]);
      const auto & cstr_l = cstr.l_leq_Ax_leq_u->l();
      const auto & cstr_u = cstr.l_leq_Ax_leq_u->u();
      for(size_t i = 0; i < 3; ++i)
      {
        FAST_CHECK_EQ(mem->A.row(i), sqrt(aW(i))*cstr_A.row(i));
        FAST_CHECK_EQ(mem->A.row(i + 3), -sqrt(aW(i))*cstr_A.row(i));
        FAST_CHECK_EQ(mem->b(i), sqrt(aW(i))*cstr_u(i));
        FAST_CHECK_EQ(mem->b(i + 3), -sqrt(aW(i))*cstr_l(i));
      }
    }

    FAST_CHECK_EQ(check(cstr.l_leq_Ax_leq_u, cstr.p0), check(*mem.get(), at.constraintType(), at.constraintRhs(), cstr.p0));
    FAST_CHECK_EQ(check(cstr.l_leq_Ax_leq_u, cstr.pl), check(*mem.get(), at.constraintType(), at.constraintRhs(), cstr.pl));
    FAST_CHECK_EQ(check(cstr.l_leq_Ax_leq_u, cstr.pu), check(*mem.get(), at.constraintType(), at.constraintRhs(), cstr.pu));
  }
}
