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
static const double large = tvm::constant::big_number;

using namespace tvm;
using namespace tvm::constraint;
using namespace tvm::scheme::internal;
using namespace tvm::requirements;
using namespace Eigen;

using BLCPtr = std::shared_ptr<BasicLinearConstraint>;

struct Constraints
{
  VectorXd p0;
  VectorXd pl;
  VectorXd pu;

  BLCPtr Ax_eq_0;
  BLCPtr Ax_geq_0;
  BLCPtr Ax_leq_0;
  BLCPtr Ax_eq_b;
  BLCPtr Ax_geq_b;
  BLCPtr Ax_leq_b;
  BLCPtr Ax_eq_minus_b;
  BLCPtr Ax_geq_minus_b;
  BLCPtr Ax_leq_minus_b;
  BLCPtr l_leq_Ax_leq_u;
  BLCPtr minus_l_leq_Ax_leq_minus_u;
};

struct Memory
{
  Memory(int m, int n) : A(m, n), b(m), l(m), u(m)
  {
    reset();
  }

  void reset()
  {
    A.setZero();
    b.setZero();
    l.setZero();
    u.setZero();
  }

  void randomize()
  {
    A.setRandom();
    b.setRandom();
    l.setRandom();
    u.setRandom();
  }

  MatrixXd A;
  VectorXd b;
  VectorXd l;
  VectorXd u;
};

//Check if the constraint is satisfied for the current value of the variable
bool check(BLCPtr c, const VectorXd& x)
{
  const double eps = 1e-12;
  c->variables()[0]->value(x);
  c->updateValue();
  auto v = c->value();
  if (c->type() == Type::DOUBLE_SIDED)
  {
    if (c->rhs() == RHS::AS_GIVEN)
      return (c->l().array()-eps <= v.array()).all() && (v.array() <= c->u().array()+eps).all();
    else
      return (-c->l().array()-eps <= v.array()).all() && (v.array() <= -c->u().array()+eps).all();
  }
  else
  {
    std::function<bool(const VectorXd&, const VectorXd&)> comp;
    const VectorXd&(BasicLinearConstraint::*rhs)() const;
    switch (c->type())
    {
    case Type::EQUAL: comp = [](const VectorXd& u, const VectorXd& v) {return u.isApprox(v); }; rhs =& BasicLinearConstraint::e;  break;
    case Type::GREATER_THAN: comp = [eps](const VectorXd& u, const VectorXd& v) {return (u.array() + eps >= v.array()).all(); }; rhs = &BasicLinearConstraint::l; break;
    case Type::LOWER_THAN: comp = [eps](const VectorXd& u, const VectorXd& v) {return (u.array() - eps <= v.array()).all(); }; rhs = &BasicLinearConstraint::u; break;
    default: break;
    }
    switch (c->rhs())
    {
    case RHS::AS_GIVEN: return comp(v, (c.get()->*rhs)()); break;
    case RHS::OPPOSITE: return comp(v, -(c.get()->*rhs)()); break;
    case RHS::ZERO: return comp(v, VectorXd::Zero(c->size())); break;
    default:
      return false;
    }
  }
}

bool check(const Memory& mem, Type ct, RHS cr, const VectorXd& x)
{
  const double eps = 1e-12;
  VectorXd v = mem.A*x;
  if (ct == Type::DOUBLE_SIDED)
  {
    if (cr == RHS::AS_GIVEN)
      return (mem.l.array() - eps <= v.array()).all() && (v.array() <= mem.u.array() + eps).all();
    else
      return (-mem.l.array() - eps <= v.array()).all() && (v.array() <= -mem.u.array() + eps).all();
  }
  else
  {
    std::function<bool(const VectorXd&, const VectorXd&)> comp;
    switch (ct)
    {
    case Type::EQUAL: comp = [](const VectorXd& u, const VectorXd& v) {return u.isApprox(v); };  break;
    case Type::GREATER_THAN: comp = [eps](const VectorXd& u, const VectorXd& v) {return (u.array() + eps >= v.array()).all(); }; break;
    case Type::LOWER_THAN: comp = [eps](const VectorXd& u, const VectorXd& v) {return (u.array() - eps <= v.array()).all(); }; break;
    default: break;
    }
    switch (cr)
    {
    case RHS::AS_GIVEN: return comp(v, mem.b); break;
    case RHS::OPPOSITE: return comp(v, -mem.b); break;
    case RHS::ZERO: return comp(v, VectorXd::Zero(mem.b.rows())); break;
    default:
      return false;
    }
  }
}

Constraints buildConstraints(int m, int n)
{
  Constraints cstr;
  VariablePtr x = Space(n).createVariable("x");

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

  cstr.Ax_eq_0 = std::make_shared<BasicLinearConstraint>(A, x, Type::EQUAL);
  cstr.Ax_geq_0 = std::make_shared<BasicLinearConstraint>(A, x, Type::GREATER_THAN);
  cstr.Ax_leq_0 = std::make_shared<BasicLinearConstraint>(A, x, Type::LOWER_THAN);

  cstr.Ax_eq_b = std::make_shared<BasicLinearConstraint>(A, x, l, Type::EQUAL);
  cstr.Ax_geq_b = std::make_shared<BasicLinearConstraint>(A, x, l, Type::GREATER_THAN);
  cstr.Ax_leq_b = std::make_shared<BasicLinearConstraint>(A, x, u, Type::LOWER_THAN);

  cstr.Ax_eq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -l, Type::EQUAL, RHS::OPPOSITE);
  cstr.Ax_geq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -l, Type::GREATER_THAN, RHS::OPPOSITE);
  cstr.Ax_leq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -u, Type::LOWER_THAN, RHS::OPPOSITE);

  cstr.l_leq_Ax_leq_u = std::make_shared<BasicLinearConstraint>(A, x, l, u);
  cstr.minus_l_leq_Ax_leq_minus_u = std::make_shared<BasicLinearConstraint>(A, x, -l, -u, RHS::OPPOSITE);
  return cstr;
}

//build constraints x op 0, x op +/-2 such that 0 is feasible, and -2 <= x <= 2
Constraints buildSimpleConstraints()
{
  Constraints cstr;
  VariablePtr x = Space(1).createVariable("x");

  //generate matrix
  MatrixXd A = MatrixXd::Identity(1, 1);
  VectorXd l = VectorXd::Constant(1, -2);
  VectorXd u = VectorXd::Constant(1, 2);

  cstr.Ax_eq_0 = std::make_shared<BasicLinearConstraint>(A, x, Type::EQUAL);
  cstr.Ax_geq_0 = std::make_shared<BasicLinearConstraint>(A, x, Type::GREATER_THAN);
  cstr.Ax_leq_0 = std::make_shared<BasicLinearConstraint>(A, x, Type::LOWER_THAN);

  cstr.Ax_eq_b = std::make_shared<BasicLinearConstraint>(A, x, l, Type::EQUAL);
  cstr.Ax_geq_b = std::make_shared<BasicLinearConstraint>(A, x, l, Type::GREATER_THAN);
  cstr.Ax_leq_b = std::make_shared<BasicLinearConstraint>(A, x, u, Type::LOWER_THAN);

  cstr.Ax_eq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -l, Type::EQUAL, RHS::OPPOSITE);
  cstr.Ax_geq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -l, Type::GREATER_THAN, RHS::OPPOSITE);
  cstr.Ax_leq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -u, Type::LOWER_THAN, RHS::OPPOSITE);

  cstr.l_leq_Ax_leq_u = std::make_shared<BasicLinearConstraint>(A, x, l, u);
  cstr.minus_l_leq_Ax_leq_minus_u = std::make_shared<BasicLinearConstraint>(A, x, -l, -u, RHS::OPPOSITE);
  return cstr;
}

//check that each point -3, -2, -1, 0, 1, 2 and 3 is either verifying at the same time
//the two descriptions of the constraint or violating both of them.
void checkSimple(BLCPtr cstr, const Memory& mem, Type t, RHS r)
{
  VectorXd pm3 = VectorXd::Constant(1, -3);
  VectorXd pm2 = VectorXd::Constant(1, -2);
  VectorXd pm1 = VectorXd::Constant(1, -1);
  VectorXd p0 = VectorXd::Constant(1, 0);
  VectorXd p1 = VectorXd::Constant(1, 1);
  VectorXd p2 = VectorXd::Constant(1, 2);
  VectorXd p3 = VectorXd::Constant(1, 3);

  FAST_CHECK_EQ(check(cstr, pm3), check(mem, t, r, pm3));
  FAST_CHECK_EQ(check(cstr, pm2), check(mem, t, r, pm2));
  FAST_CHECK_EQ(check(cstr, pm1), check(mem, t, r, pm1));
  FAST_CHECK_EQ(check(cstr, p0), check(mem, t, r, p0));
  FAST_CHECK_EQ(check(cstr, p1), check(mem, t, r, p1));
  FAST_CHECK_EQ(check(cstr, p3), check(mem, t, r, p3));
}

void checkAssignment(BLCPtr c, const AssignmentTarget& at, Memory& mem, Type t, RHS r, bool throws)
{
  auto req = std::make_shared<SolvingRequirements>();
  VariableVector vars(c->variables());
  mem.randomize();
  if (throws)
  {
    CHECK_THROWS(Assignment a(c, req, at, vars));
  }
  else
  {
    Assignment a(c, req, at, vars);
    a.run();
    checkSimple(c, mem, t, r);
  }
}

/** Check the assignement of a constraint \p c to a non double-sided constraint
  * \p throws is a vector of 11 bool indicating if the assignement construction is
  * expected to throw. The order of the targets conventions are:
  * Cx=0, Cx=d, Cx=-d, Cx>=0, Cx>=d, Cx>=-d, Cx<=0, Cx<=d, Cx<=-d, l<=Cx<=u, -l<=Cx<=-u
  */
void checkSimple(BLCPtr c, std::vector<bool> throws)
{
  int s = c->type() == Type::DOUBLE_SIDED ? 2 : 1;
  auto sMem = std::make_shared<Memory>(s, 1);
  auto sRange = std::make_shared<Range>(0, s);
  auto dMem = std::make_shared<Memory>(1, 1);
  auto dRange = std::make_shared<Range>(0, 1);

  // target Cx=0
  {
    auto t = Type::EQUAL;
    auto r = RHS::ZERO;
    AssignmentTarget at(sRange, sMem->A, t);
    checkAssignment(c, at, *sMem, t, r, throws[0]);
  }
  // target Cx=d
  {
    auto t = Type::EQUAL;
    auto r = RHS::AS_GIVEN;
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkAssignment(c, at, *sMem, t, r, throws[1]);
  }
  // target Cx=-d
  {
    auto t = Type::EQUAL;
    auto r = RHS::OPPOSITE;
    sMem->randomize();
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkAssignment(c, at, *sMem, t, r, throws[2]);
  }
  // target Cx>=0
  {
    auto t = Type::GREATER_THAN;
    auto r = RHS::ZERO;
    AssignmentTarget at(sRange, sMem->A, t);
    checkAssignment(c, at, *sMem, t, r, throws[3]);
  }
  // target Cx>=d
  {
    auto t = Type::GREATER_THAN;
    auto r = RHS::AS_GIVEN;
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkAssignment(c, at, *sMem, t, r, throws[4]);
  }
  // target Cx>=-d
  {
    auto t = Type::GREATER_THAN;
    auto r = RHS::OPPOSITE;
    sMem->randomize();
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkAssignment(c, at, *sMem, t, r, throws[5]);
  }
  // target Cx<=0
  {
    auto t = Type::LOWER_THAN;
    auto r = RHS::ZERO;
    AssignmentTarget at(sRange, sMem->A, t);
    checkAssignment(c, at, *sMem, t, r, throws[6]);
  }
  // target Cx<=d
  {
    auto t = Type::LOWER_THAN;
    auto r = RHS::AS_GIVEN;
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkAssignment(c, at, *sMem, t, r, throws[7]);
  }
  // target Cx<=-d
  {
    auto t = Type::LOWER_THAN;
    auto r = RHS::OPPOSITE;
    sMem->randomize();
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkAssignment(c, at, *sMem, t, r, throws[8]);
  }
  // target l<=Cx<=u
  {
    auto t = Type::DOUBLE_SIDED;
    auto r = RHS::AS_GIVEN;
    AssignmentTarget at(dRange, dMem->A, dMem->l, dMem->u, r);
    checkAssignment(c, at, *dMem, t, r, throws[9]);
  }
  // target -l<=Cx<=-u
  {
    auto t = Type::DOUBLE_SIDED;
    auto r = RHS::AS_GIVEN;
    AssignmentTarget at(dRange, dMem->A, dMem->l, dMem->u, r);
    checkAssignment(c, at, *dMem, t, r, throws[10]);
  }
}

//test for correct signs on the matrices and vector
TEST_CASE("Test simple assignment")
{
  Constraints cstr = buildSimpleConstraints();
  using T = std::vector<bool>;
  const bool t = true;
  const bool f = false;
  // constraint Ax = 0
  {
    //     Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v = { f,      f,      f,      t,      t,      t,      t,      t,      t,      f,      f };
    checkSimple(cstr.Ax_eq_0, v);
  }
  // constraint Ax >= 0
  {
    //     Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v = { t,      t,      t,      f,      f,      f,      f,      f,      f,      f,      f };
    checkSimple(cstr.Ax_geq_0, v);
  }
  // constraint Ax <= 0
  {
    //     Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v = { t,      t,      t,      f,      f,      f,      f,      f,      f,      f,      f };
    checkSimple(cstr.Ax_leq_0, v);
  }
  // constraint Ax = d
  {
    //     Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v = { t,      f,      f,      t,      t,      t,      t,      t,      t,      f,      f };
    checkSimple(cstr.Ax_eq_b, v);
  }
  // constraint Ax >= d
  {
    //     Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    checkSimple(cstr.Ax_geq_b, v);
  }
  // constraint Ax <= d
  {
    //     Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    checkSimple(cstr.Ax_leq_b, v);
  }
  // constraint Ax = -d
  {
    //     Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v = { t,      f,      f,      t,      t,      t,      t,      t,      t,      f,      f };
    checkSimple(cstr.Ax_eq_minus_b, v);
  }
  // constraint Ax >= -d
  {
    //     Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    checkSimple(cstr.Ax_geq_minus_b, v);
  }
  // constraint Ax <= -d
  {
    //     Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    checkSimple(cstr.Ax_leq_minus_b, v);
  }
  // constraint l <= Ax <= u
  {
    //     Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    checkSimple(cstr.l_leq_Ax_leq_u, v);
  }
  // constraint -l <= Ax <= -u
  {
    //     Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    checkSimple(cstr.l_leq_Ax_leq_u, v);
  }
}

TEST_CASE("Test assigments")
{
  Constraints cstr = buildConstraints(3, 7);

  {
    auto mem = std::make_shared<Memory>(6, 7);
    //assignment to a target with convention l <= Ax <= u, from convention Ax >= -b
    auto range = std::make_shared<Range>(2, 3);
    AssignmentTarget at(range, mem->A, mem->l, mem->u , RHS::AS_GIVEN);
    auto req = std::make_shared<SolvingRequirements>(Weight(2.));
    VariableVector vv(cstr.Ax_eq_0->variables());
    Assignment a(cstr.Ax_geq_minus_b, req, at, vv);
    a.run();

    {
      const auto & cstr_A = cstr.Ax_geq_minus_b->jacobian(*cstr.Ax_geq_minus_b->variables()[0]);
      const auto & cstr_l = cstr.Ax_geq_minus_b->l();
      FAST_CHECK_EQ(mem->A.block(range->start, 0, 3, 7), sqrt(2)*cstr_A);
      FAST_CHECK_EQ(mem->l.block(range->start, 0, 3, 1), -sqrt(2)*cstr_l);
      FAST_CHECK_EQ(mem->u.block(range->start, 0, 3, 1), sqrt(2)*VectorXd(3).setConstant(large));
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
      FAST_CHECK_EQ(mem->u.block(range->start, 0, 3, 1), sqrt(2)*VectorXd(3).setConstant(large));
    }
  }

  {
    auto mem = std::make_shared<Memory>(6, 7);
    //assignment to a target with convention Ax <= b, from convention l <= Ax <= u
    auto range = std::make_shared<Range>(0, 6); //we need double range
    AssignmentTarget at(range, mem->A, mem->b, Type::LOWER_THAN, RHS::AS_GIVEN);
    Vector3d aW = {1., 2., 3.};
    auto req = std::make_shared<SolvingRequirements>(AnisotropicWeight{ aW });
    VariableVector vv(cstr.Ax_eq_0->variables());
    Assignment a(cstr.l_leq_Ax_leq_u, req, at, vv);
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
