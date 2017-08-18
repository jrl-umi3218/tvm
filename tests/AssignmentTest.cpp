#include <functional>
#include <iostream>

#include "Assignment.h"
#include "AssignmentTarget.h"
#include "LinearConstraint.h"
#include "Variable.h"
#include "VariableVector.h"

#include <Eigen/Core>
#include <Eigen/QR>

// boost
#define BOOST_TEST_MODULE AssignmentTest
#include <boost/test/unit_test.hpp>

using namespace Eigen;

using namespace tvm;
using namespace tvm::utils;

//FIXME see src/Assignment.cpp
static const double large = 1e6;

struct Constraints
{
  Eigen::VectorXd p0;
  Eigen::VectorXd pl;
  Eigen::VectorXd pu;

  std::shared_ptr<BasicLinearConstraint> Ax_eq_0;
  std::shared_ptr<BasicLinearConstraint> Ax_geq_0;
  std::shared_ptr<BasicLinearConstraint> Ax_leq_0;
  std::shared_ptr<BasicLinearConstraint> Ax_eq_b;
  std::shared_ptr<BasicLinearConstraint> Ax_geq_b;
  std::shared_ptr<BasicLinearConstraint> Ax_leq_b;
  std::shared_ptr<BasicLinearConstraint> Ax_eq_minus_b;
  std::shared_ptr<BasicLinearConstraint> Ax_geq_minus_b;
  std::shared_ptr<BasicLinearConstraint> Ax_leq_minus_b;
  std::shared_ptr<BasicLinearConstraint> l_leq_Ax_leq_u;
  std::shared_ptr<BasicLinearConstraint> minus_l_leq_Ax_leq_minus_u;
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

  MatrixXd A;
  VectorXd b;
  VectorXd l;
  VectorXd u;
};

//Check if the constraint is satisfied for the current value of the variable
bool check(std::shared_ptr<BasicLinearConstraint> c, const VectorXd& x)
{
  const double eps = 1e-12;
  c->variables()[0]->value(x);
  c->updateValue();
  auto v = c->value();
  if (c->constraintType() == ConstraintType::DOUBLE_SIDED)
  {
    if (c->rhsType() == RHSType::AS_GIVEN)
      return (c->l().array()-eps <= v.array()).all() && (v.array() <= c->u().array()+eps).all();
    else
      return (-c->l().array()-eps <= v.array()).all() && (v.array() <= -c->u().array()+eps).all();
  }
  else
  {
    std::function<bool(const VectorXd&, const VectorXd&)> comp;
    const VectorXd&(BasicLinearConstraint::*rhs)() const;
    switch (c->constraintType())
    {
    case ConstraintType::EQUAL: comp = [](const VectorXd& u, const VectorXd& v) {return u.isApprox(v); }; rhs =& BasicLinearConstraint::e;  break;
    case ConstraintType::GREATER_THAN: comp = [eps](const VectorXd& u, const VectorXd& v) {return (u.array() + eps >= v.array()).all(); }; rhs = &BasicLinearConstraint::l; break;
    case ConstraintType::LOWER_THAN: comp = [eps](const VectorXd& u, const VectorXd& v) {return (u.array() - eps <= v.array()).all(); }; rhs = &BasicLinearConstraint::u; break;
    default: break;
    }
    switch (c->rhsType())
    {
    case RHSType::AS_GIVEN: return comp(v, (c.get()->*rhs)()); break;
    case RHSType::OPPOSITE: return comp(v, -(c.get()->*rhs)()); break;
    case RHSType::ZERO: return comp(v, VectorXd::Zero(c->size())); break;
    default:
      return false;
    }
  }
}

bool check(const Memory& mem, ConstraintType ct, RHSType rt, const VectorXd& x)
{
  const double eps = 1e-12;
  VectorXd v = mem.A*x;
  if (ct == ConstraintType::DOUBLE_SIDED)
  {
    if (rt == RHSType::AS_GIVEN)
      return (mem.l.array() - eps <= v.array()).all() && (v.array() <= mem.u.array() + eps).all();
    else
      return (-mem.l.array() - eps <= v.array()).all() && (v.array() <= -mem.u.array() + eps).all();
  }
  else
  {
    std::function<bool(const VectorXd&, const VectorXd&)> comp;
    switch (ct)
    {
    case ConstraintType::EQUAL: comp = [](const VectorXd& u, const VectorXd& v) {return u.isApprox(v); };  break;
    case ConstraintType::GREATER_THAN: comp = [eps](const VectorXd& u, const VectorXd& v) {return (u.array() + eps >= v.array()).all(); }; break;
    case ConstraintType::LOWER_THAN: comp = [eps](const VectorXd& u, const VectorXd& v) {return (u.array() - eps <= v.array()).all(); }; break;
    default: break;
    }
    switch (rt)
    {
    case RHSType::AS_GIVEN: return comp(v, mem.b); break;
    case RHSType::OPPOSITE: return comp(v, -mem.b); break;
    case RHSType::ZERO: return comp(v, VectorXd::Zero(mem.b.rows())); break;
    default:
      return false;
    }
  }
}

Constraints buildConstraints(int m, int n)
{
  Constraints cstr;
  std::shared_ptr<Variable> x = Space(n).createVariable("x");

  //generate matrix
  MatrixXd A = MatrixXd::Random(m, n);
  VectorXd l = -VectorXd::Random(m).cwiseAbs();
  VectorXd u = VectorXd::Random(m).cwiseAbs();

  //Point p0 such that Ap0 = 0
  cstr.p0 = A.householderQr().solve(VectorXd::Zero(m));
  //Point pl such that Apl = l
  cstr.pl = A.householderQr().solve(l);
  //Point pu such that Apu = u
  cstr.pu = A.householderQr().solve(u);

  cstr.Ax_eq_0 = std::make_shared<BasicLinearConstraint>(A, x, ConstraintType::EQUAL);
  cstr.Ax_geq_0 = std::make_shared<BasicLinearConstraint>(A, x, ConstraintType::GREATER_THAN);
  cstr.Ax_leq_0 = std::make_shared<BasicLinearConstraint>(A, x, ConstraintType::LOWER_THAN);

  cstr.Ax_eq_b = std::make_shared<BasicLinearConstraint>(A, x, l, ConstraintType::EQUAL);
  cstr.Ax_geq_b = std::make_shared<BasicLinearConstraint>(A, x, l, ConstraintType::GREATER_THAN);
  cstr.Ax_leq_b = std::make_shared<BasicLinearConstraint>(A, x, u, ConstraintType::LOWER_THAN);

  cstr.Ax_eq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -l, ConstraintType::EQUAL, RHSType::OPPOSITE);
  cstr.Ax_geq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -l, ConstraintType::GREATER_THAN, RHSType::OPPOSITE);
  cstr.Ax_leq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -u, ConstraintType::LOWER_THAN, RHSType::OPPOSITE);

  cstr.l_leq_Ax_leq_u = std::make_shared<BasicLinearConstraint>(A, x, l, u);
  cstr.minus_l_leq_Ax_leq_minus_u = std::make_shared<BasicLinearConstraint>(A, x, -l, -u, RHSType::OPPOSITE);
  return cstr;
}

BOOST_AUTO_TEST_CASE(AssignmentTest)
{
  Constraints cstr = buildConstraints(3, 7);

  {
    auto mem = std::make_shared<Memory>(6, 7);
    //assignment to a target with convention l <= Ax <= u, from convention Ax >= -b
    auto range = std::make_shared<Range>(2, 3);
    AssignmentTarget at(range, { mem, &mem->A }, { mem, &mem->l }, { mem, &mem->u }, RHSType::AS_GIVEN);
    SolvingRequirements req({ Weight(2.) });
    VariableVector vv(cstr.Ax_eq_0->variables());
    Assignment a(cstr.Ax_geq_minus_b, req, at, vv);
    a.run();

    {
      const auto & cstr_A = cstr.Ax_geq_minus_b->jacobian(*cstr.Ax_geq_minus_b->variables()[0]);
      const auto & cstr_l = cstr.Ax_geq_minus_b->lNoCheck();
      BOOST_CHECK(mem->A.block(range->start, 0, 3, 7) == sqrt(2)*cstr_A);
      BOOST_CHECK(mem->l.block(range->start, 0, 3, 1) == -sqrt(2)*cstr_l);
      BOOST_CHECK(mem->u.block(range->start, 0, 3, 1) == sqrt(2)*Eigen::VectorXd(3).setConstant(large));
    }

    BOOST_CHECK(check(cstr.Ax_geq_minus_b, cstr.p0) == check(*mem.get(), at.constraintType(), at.rhsType(), cstr.p0));
    BOOST_CHECK(check(cstr.Ax_geq_minus_b, cstr.pl) == check(*mem.get(), at.constraintType(), at.rhsType(), cstr.pl));
    BOOST_CHECK(check(cstr.Ax_geq_minus_b, cstr.pu) == check(*mem.get(), at.constraintType(), at.rhsType(), cstr.pu));

    //now we change the range of the target and refresh the assignment
    range->start = 0;
    a.refreshTarget();
    mem->A.setZero();
    mem->l.setZero();
    mem->u.setZero();
    a.run();

    {
      const auto & cstr_A = cstr.Ax_geq_minus_b->jacobian(*cstr.Ax_geq_minus_b->variables()[0]);
      const auto & cstr_l = cstr.Ax_geq_minus_b->lNoCheck();
      BOOST_CHECK(mem->A.block(range->start, 0, 3, 7) == sqrt(2)*cstr_A);
      BOOST_CHECK(mem->l.block(range->start, 0, 3, 1) == -sqrt(2)*cstr_l);
      BOOST_CHECK(mem->u.block(range->start, 0, 3, 1) == sqrt(2)*Eigen::VectorXd(3).setConstant(large));
    }
  }

  {
    auto mem = std::make_shared<Memory>(6, 7);
    //assignment to a target with convention Ax <= b, from convention l <= Ax <= u
    auto range = std::make_shared<Range>(0, 6); //we need double range
    AssignmentTarget at(range, { mem, &mem->A }, { mem, &mem->b }, ConstraintType::LOWER_THAN, RHSType::AS_GIVEN);
    Eigen::Vector3d aW = {1., 2., 3.};
    SolvingRequirements req({ AnisotropicWeight(aW) });
    VariableVector vv(cstr.Ax_eq_0->variables());
    Assignment a(cstr.l_leq_Ax_leq_u, req, at, vv);
    a.run();

    {
      const auto & cstr_A = cstr.l_leq_Ax_leq_u->jacobian(*cstr.l_leq_Ax_leq_u->variables()[0]);
      const auto & cstr_l = cstr.l_leq_Ax_leq_u->lNoCheck();
      const auto & cstr_u = cstr.l_leq_Ax_leq_u->uNoCheck();
      for(size_t i = 0; i < 3; ++i)
      {
        BOOST_CHECK(mem->A.row(i) == sqrt(aW(i))*cstr_A.row(i));
        BOOST_CHECK(mem->A.row(i + 3) == -sqrt(aW(i))*cstr_A.row(i));
        BOOST_CHECK(mem->b(i) == sqrt(aW(i))*cstr_u(i));
        BOOST_CHECK(mem->b(i + 3) == -sqrt(aW(i))*cstr_l(i));
      }
    }

    BOOST_CHECK(check(cstr.l_leq_Ax_leq_u, cstr.p0) == check(*mem.get(), at.constraintType(), at.rhsType(), cstr.p0));
    BOOST_CHECK(check(cstr.l_leq_Ax_leq_u, cstr.pl) == check(*mem.get(), at.constraintType(), at.rhsType(), cstr.pl));
    BOOST_CHECK(check(cstr.l_leq_Ax_leq_u, cstr.pu) == check(*mem.get(), at.constraintType(), at.rhsType(), cstr.pu));
  }
}
