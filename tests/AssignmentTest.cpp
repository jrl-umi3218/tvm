/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <functional>
#include <iostream>
#include <memory>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <tvm/Variable.h>
#include <tvm/VariableVector.h>
#include <tvm/constraint/BasicLinearConstraint.h>
#include <tvm/hint/internal/Substitutions.h>
#include <tvm/hint/Substitution.h>
#include <tvm/scheme/internal/Assignment.h>
#include <tvm/scheme/internal/AssignmentTarget.h>

#include <Eigen/Core>
#include <Eigen/QR>

//FIXME see src/Assignment.cpp
static const double large = tvm::constant::big_number;

using namespace tvm;
using namespace tvm::constraint;
using namespace tvm::hint;
using namespace tvm::hint::internal;
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
  const_cast<VariableVector&>(c->variables()).value(x);
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

bool check(const Memory& mem, Type ct, RHS cr, const VectorXd& x, bool bound=false)
{
  const double eps = 1e-12;
  VectorXd v;
  if (bound)
    v = x;
  else
    v = mem.A*x;

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
//if minus = true, we take -x instead of x
Constraints buildSimpleConstraints(double s = 1, VariablePtr y = nullptr)
{
  assert(s != 0);
  Constraints cstr;
  VariablePtr x;
  if (y)
    x = y;
  else
    x = Space(1).createVariable("x");

  tvm::internal::MatrixProperties p;
  if (s == -1)
    p = { tvm::internal::MatrixProperties::MINUS_IDENTITY };
  else if (s == 1)
    p = { tvm::internal::MatrixProperties::IDENTITY };
  else
    p = { tvm::internal::MatrixProperties::MULTIPLE_OF_IDENTITY, 
          tvm::internal::MatrixProperties::Invertibility(s != 0) };

  //generate matrix
  MatrixXd A = s*MatrixXd::Identity(1, 1);
  VectorXd l = VectorXd::Constant(1, -2);
  VectorXd u = VectorXd::Constant(1, 2);

  cstr.Ax_eq_0 = std::make_shared<BasicLinearConstraint>(A, x, Type::EQUAL);
  cstr.Ax_eq_0->A(A, p);
  cstr.Ax_geq_0 = std::make_shared<BasicLinearConstraint>(A, x, Type::GREATER_THAN);
  cstr.Ax_geq_0->A(A, p);
  cstr.Ax_leq_0 = std::make_shared<BasicLinearConstraint>(A, x, Type::LOWER_THAN);
  cstr.Ax_leq_0->A(A, p);

  cstr.Ax_eq_b = std::make_shared<BasicLinearConstraint>(A, x, l, Type::EQUAL);
  cstr.Ax_eq_b->A(A, p);
  cstr.Ax_geq_b = std::make_shared<BasicLinearConstraint>(A, x, l, Type::GREATER_THAN);
  cstr.Ax_geq_b->A(A, p);
  cstr.Ax_leq_b = std::make_shared<BasicLinearConstraint>(A, x, u, Type::LOWER_THAN);
  cstr.Ax_leq_b->A(A, p);

  cstr.Ax_eq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -l, Type::EQUAL, RHS::OPPOSITE);
  cstr.Ax_eq_minus_b->A(A, p);
  cstr.Ax_geq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -l, Type::GREATER_THAN, RHS::OPPOSITE);
  cstr.Ax_geq_minus_b->A(A, p);
  cstr.Ax_leq_minus_b = std::make_shared<BasicLinearConstraint>(A, x, -u, Type::LOWER_THAN, RHS::OPPOSITE);
  cstr.Ax_leq_minus_b->A(A, p);

  cstr.l_leq_Ax_leq_u = std::make_shared<BasicLinearConstraint>(A, x, l, u);
  cstr.l_leq_Ax_leq_u->A(A, p);
  cstr.minus_l_leq_Ax_leq_minus_u = std::make_shared<BasicLinearConstraint>(A, x, -l, -u, RHS::OPPOSITE);
  cstr.minus_l_leq_Ax_leq_minus_u->A(A, p);
  return cstr;
}

//create a set of constraints -3x+4y op +/-2 and a constraint for substitution
// -x + 2y - z = 0/1/-1 (the latter value depending on the rhs param).
std::pair<Constraints, BLCPtr> buildSimpleSubstitution(RHS rhs)
{
  Constraints cstr;
  VariablePtr x = Space(1).createVariable("x");
  VariablePtr y = Space(1).createVariable("y");
  VariablePtr z = Space(1).createVariable("z");

  //generate matrix
  MatrixXd I = MatrixXd::Identity(1, 1);
  MatrixXd Ax = -3 * I;
  MatrixXd Ay = 4 * I;
  VectorXd l = VectorXd::Constant(1, -2);
  VectorXd u = VectorXd::Constant(1, 2);

  std::vector<Ref<const MatrixXd>> A = { Ax, Ay };
  std::vector<VariablePtr> v = { x, y };

  cstr.Ax_eq_0 = std::make_shared<BasicLinearConstraint>( A, v, Type::EQUAL);
  cstr.Ax_geq_0 = std::make_shared<BasicLinearConstraint>(A, v, Type::GREATER_THAN);
  cstr.Ax_leq_0 = std::make_shared<BasicLinearConstraint>(A, v, Type::LOWER_THAN);

  cstr.Ax_eq_b = std::make_shared<BasicLinearConstraint>( A, v, l, Type::EQUAL);
  cstr.Ax_geq_b = std::make_shared<BasicLinearConstraint>(A, v, l, Type::GREATER_THAN);
  cstr.Ax_leq_b = std::make_shared<BasicLinearConstraint>(A, v, u, Type::LOWER_THAN);

  cstr.Ax_eq_minus_b = std::make_shared<BasicLinearConstraint>( A, v, -l, Type::EQUAL, RHS::OPPOSITE);
  cstr.Ax_geq_minus_b = std::make_shared<BasicLinearConstraint>(A, v, -l, Type::GREATER_THAN, RHS::OPPOSITE);
  cstr.Ax_leq_minus_b = std::make_shared<BasicLinearConstraint>(A, v, -u, Type::LOWER_THAN, RHS::OPPOSITE);

  cstr.l_leq_Ax_leq_u = std::make_shared<BasicLinearConstraint>(A, v, l, u);
  cstr.minus_l_leq_Ax_leq_minus_u = std::make_shared<BasicLinearConstraint>(A, v, -l, -u, RHS::OPPOSITE);

  MatrixXd Cx = -I;
  MatrixXd Cy = 2 * I;
  MatrixXd Cz = -I;
  std::vector<Ref<const MatrixXd>> C = { Cx, Cy, Cz };
  std::vector<VariablePtr> w = { x, y, z };
  VectorXd d = VectorXd::Constant(1, 1);
  if (rhs == RHS::ZERO)
  {
    return std::make_pair(cstr, std::make_shared<BasicLinearConstraint>(C, w, Type::EQUAL));
  }
  else
  {
    return std::make_pair(cstr, std::make_shared<BasicLinearConstraint>(C, w, d, Type::EQUAL, rhs));
  }
}

//check that each point -3, -2, -1, 0, 1, 2 and 3 is either verifying at the same time
//the two descriptions of the constraint or violating both of them.
void checkSimple(BLCPtr cstr, const Memory& mem, Type t, RHS r, bool bound = false)
{
  VectorXd pm3 = VectorXd::Constant(1, -3);
  VectorXd pm2 = VectorXd::Constant(1, -2);
  VectorXd pm1 = VectorXd::Constant(1, -1);
  VectorXd p0 = VectorXd::Constant(1, 0);
  VectorXd p1 = VectorXd::Constant(1, 1);
  VectorXd p2 = VectorXd::Constant(1, 2);
  VectorXd p3 = VectorXd::Constant(1, 3);

  FAST_CHECK_EQ(check(cstr, pm3), check(mem, t, r, pm3, bound));
  FAST_CHECK_EQ(check(cstr, pm2), check(mem, t, r, pm2, bound));
  FAST_CHECK_EQ(check(cstr, pm1), check(mem, t, r, pm1, bound));
  FAST_CHECK_EQ(check(cstr, p0), check(mem, t, r, p0, bound));
  FAST_CHECK_EQ(check(cstr, p1), check(mem, t, r, p1, bound));
  FAST_CHECK_EQ(check(cstr, p2), check(mem, t, r, p2, bound));
  FAST_CHECK_EQ(check(cstr, p3), check(mem, t, r, p3, bound));
}

//check for the intersection of 2 bound constraints
void checkSimple(BLCPtr cstr1, BLCPtr cstr2, const Memory& mem)
{
  Type t = Type::DOUBLE_SIDED;
  RHS r = RHS::AS_GIVEN;

  VectorXd pm3 = VectorXd::Constant(1, -3);
  VectorXd pm2 = VectorXd::Constant(1, -2);
  VectorXd pm1 = VectorXd::Constant(1, -1);
  VectorXd p0 = VectorXd::Constant(1, 0);
  VectorXd p1 = VectorXd::Constant(1, 1);
  VectorXd p2 = VectorXd::Constant(1, 2);
  VectorXd p3 = VectorXd::Constant(1, 3);

  FAST_CHECK_EQ(check(cstr1, pm3) && check(cstr2, pm3), check(mem, t, r, pm3, true));
  FAST_CHECK_EQ(check(cstr1, pm2) && check(cstr2, pm2), check(mem, t, r, pm2, true));
  FAST_CHECK_EQ(check(cstr1, pm1) && check(cstr2, pm1), check(mem, t, r, pm1, true));
  FAST_CHECK_EQ(check(cstr1, p0) && check(cstr2, p0), check(mem, t, r, p0, true));
  FAST_CHECK_EQ(check(cstr1, p1) && check(cstr2, p1), check(mem, t, r, p1, true));
  FAST_CHECK_EQ(check(cstr1, p2) && check(cstr2, p2), check(mem, t, r, p2, true));
  FAST_CHECK_EQ(check(cstr1, p3) && check(cstr2, p3), check(mem, t, r, p3, true));
}

// Check that each point (x,z) with z=-x, for x = -5,-3,1,0,1,3,5 verifies or violates the
// substituted constraint in mem at the same time as the corresponding (x, y) point verifies
// or violates the constraint.
void checkSubstitution(BLCPtr cstr, RHS subRhs, const Memory& mem, Type t, RHS r)
{
  double y;
  switch (subRhs)
  {
  case RHS::ZERO: y = 0; break;
  case RHS::AS_GIVEN: y = 0.5; break;
  case RHS::OPPOSITE: y = -0.5; break;
  }
  VectorXd xym5(2); xym5 << -2, y;
  VectorXd xym3(2); xym3 << -4./3, y;
  VectorXd xym1(2); xym1 << -2./3, y;
  VectorXd xy0(2); xy0 << 0, y;
  VectorXd xy1(2); xy1 << 2./3, y;
  VectorXd xy3(2); xy3 << 4./3, y;
  VectorXd xy5(2); xy5 << 2, y;
  VectorXd xzm5(2); xzm5 << -2, 2;
  VectorXd xzm3(2); xzm3 << -4./3, 4./3;
  VectorXd xzm1(2); xzm1 << -2./3, 2./3;
  VectorXd xz0(2); xz0 << 0, 0;
  VectorXd xz1(2); xz1 << 2./3, -2./3;
  VectorXd xz3(2); xz3 << 4./3, -4./3;
  VectorXd xz5(2); xz5 << 2, -2;

  //std::cout << check(cstr, xym5) <<", " << check(mem, t, r, xzm5, false) << std::endl;
  //std::cout << check(cstr, xym3) <<", " << check(mem, t, r, xzm3, false) << std::endl;
  //std::cout << check(cstr, xym1) <<", " << check(mem, t, r, xzm1, false) << std::endl;
  //std::cout << check(cstr, xy0)  <<", " << check(mem, t, r, xz0, false)  << std::endl;
  //std::cout << check(cstr, xy1)  <<", " << check(mem, t, r, xz1, false)  << std::endl;
  //std::cout << check(cstr, xy3)  <<", " << check(mem, t, r, xz3, false)  << std::endl;
  //std::cout << check(cstr, xy5)  <<", " << check(mem, t, r, xz5, false)  << std::endl;

  FAST_CHECK_EQ(check(cstr, xym5), check(mem, t, r, xzm5, false));
  FAST_CHECK_EQ(check(cstr, xym3), check(mem, t, r, xzm3, false));
  FAST_CHECK_EQ(check(cstr, xym1), check(mem, t, r, xzm1, false));
  FAST_CHECK_EQ(check(cstr, xy0), check(mem, t, r, xz0, false));
  FAST_CHECK_EQ(check(cstr, xy1), check(mem, t, r, xz1, false));
  FAST_CHECK_EQ(check(cstr, xy3), check(mem, t, r, xz3, false));
  FAST_CHECK_EQ(check(cstr, xy5), check(mem, t, r, xz5, false));
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

void checkSubstitutionAssignment(BLCPtr c, Substitutions& s, const AssignmentTarget& at, Memory& mem, Type t, RHS r, bool throws)
{
  auto req = std::make_shared<SolvingRequirements>();
  VariableVector vars;
  vars.add(s.otherVariables());
  mem.randomize();
  if (throws)
  {
    CHECK_THROWS(Assignment a(c, req, at, vars, &s));
  }
  else
  {
    s.updateSubstitutions();
    Assignment a(c, req, at, vars, &s);
    a.run();
    checkSubstitution(c, s.substitutions()[0].constraints()[0]->rhs(), mem, t, r);
  }
}

void checkBoundAssignment(BLCPtr c, const AssignmentTarget& at, Memory& mem)
{
  Assignment a(c, at, c->variables()[0], true);
  a.run();
  checkSimple(c, mem, Type::DOUBLE_SIDED, RHS::AS_GIVEN, true);
}

void checkBoundAssignment(BLCPtr c1, BLCPtr c2, const AssignmentTarget& at, Memory& mem)
{
  assert(c1->variables()[0] == c2->variables()[0]);
  Assignment a1(c1, at, c1->variables()[0], true);
  a1.run();
  Assignment a2(c2, at, c2->variables()[0], false);
  a2.run();
  checkSimple(c1, c2, mem);
}

/** Check the assignement of a constraint \p c to a target
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

//same as above but introducing a substitution
void checkSimple(BLCPtr c, const Substitution& sub, std::vector<bool> throws)
{
  int s = c->type() == Type::DOUBLE_SIDED ? 2 : 1;
  auto sMem = std::make_shared<Memory>(s, 2);
  auto sRange = std::make_shared<Range>(0, s);
  auto dMem = std::make_shared<Memory>(1, 2);
  auto dRange = std::make_shared<Range>(0, 1);

  Substitutions subs;
  subs.add(sub);
  subs.finalize();

  // target Cx=0
  {
    auto t = Type::EQUAL;
    auto r = RHS::ZERO;
    AssignmentTarget at(sRange, sMem->A, t);
    checkSubstitutionAssignment(c, subs, at, *sMem, t, r, throws[0]);
  }
  // target Cx=d
  {
    auto t = Type::EQUAL;
    auto r = RHS::AS_GIVEN;
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkSubstitutionAssignment(c, subs, at, *sMem, t, r, throws[1]);
  }
  // target Cx=-d
  {
    auto t = Type::EQUAL;
    auto r = RHS::OPPOSITE;
    sMem->randomize();
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkSubstitutionAssignment(c, subs, at, *sMem, t, r, throws[2]);
  }
  // target Cx>=0
  {
    auto t = Type::GREATER_THAN;
    auto r = RHS::ZERO;
    AssignmentTarget at(sRange, sMem->A, t);
    checkSubstitutionAssignment(c, subs, at, *sMem, t, r, throws[3]);
  }
  // target Cx>=d
  {
    auto t = Type::GREATER_THAN;
    auto r = RHS::AS_GIVEN;
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkSubstitutionAssignment(c, subs, at, *sMem, t, r, throws[4]);
  }
  // target Cx>=-d
  {
    auto t = Type::GREATER_THAN;
    auto r = RHS::OPPOSITE;
    sMem->randomize();
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkSubstitutionAssignment(c, subs, at, *sMem, t, r, throws[5]);
  }
  // target Cx<=0
  {
    auto t = Type::LOWER_THAN;
    auto r = RHS::ZERO;
    AssignmentTarget at(sRange, sMem->A, t);
    checkSubstitutionAssignment(c, subs, at, *sMem, t, r, throws[6]);
  }
  // target Cx<=d
  {
    auto t = Type::LOWER_THAN;
    auto r = RHS::AS_GIVEN;
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkSubstitutionAssignment(c, subs, at, *sMem, t, r, throws[7]);
  }
  // target Cx<=-d
  {
    auto t = Type::LOWER_THAN;
    auto r = RHS::OPPOSITE;
    sMem->randomize();
    AssignmentTarget at(sRange, sMem->A, sMem->b, t, r);
    checkSubstitutionAssignment(c, subs, at, *sMem, t, r, throws[8]);
  }
  // target l<=Cx<=u
  {
    auto t = Type::DOUBLE_SIDED;
    auto r = RHS::AS_GIVEN;
    AssignmentTarget at(dRange, dMem->A, dMem->l, dMem->u, r);
    checkSubstitutionAssignment(c, subs, at, *dMem, t, r, throws[9]);
  }
  // target -l<=Cx<=-u
  {
    auto t = Type::DOUBLE_SIDED;
    auto r = RHS::AS_GIVEN;
    AssignmentTarget at(dRange, dMem->A, dMem->l, dMem->u, r);
    checkSubstitutionAssignment(c, subs, at, *dMem, t, r, throws[10]);
  }
}

/** Check the assignement of a bound \p c to a target */
void checkSimpleBound(BLCPtr c)
{
  auto dMem = std::make_shared<Memory>(1, 1);
  auto dRange = std::make_shared<Range>(0, 1);
  
  // target l<=x<=u
  AssignmentTarget at(dRange, dMem->l, dMem->u);
  checkBoundAssignment(c, at, *dMem);
}

void checkSimpleBound(BLCPtr c1, BLCPtr c2)
{
  auto dMem = std::make_shared<Memory>(1, 1);
  auto dRange = std::make_shared<Range>(0, 1);

  // target l<=x<=u
  AssignmentTarget at(dRange, dMem->l, dMem->u);
  checkBoundAssignment(c1, c2, at, *dMem);
}

// test for correct signs on the matrices and vector
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

  //now the bounds
  checkSimpleBound(cstr.Ax_eq_0);
  checkSimpleBound(cstr.Ax_geq_0);
  checkSimpleBound(cstr.Ax_leq_0);
  checkSimpleBound(cstr.Ax_eq_b);
  checkSimpleBound(cstr.Ax_geq_b);
  checkSimpleBound(cstr.Ax_leq_b);
  checkSimpleBound(cstr.Ax_eq_minus_b);
  checkSimpleBound(cstr.Ax_geq_minus_b);
  checkSimpleBound(cstr.Ax_leq_minus_b);
  checkSimpleBound(cstr.l_leq_Ax_leq_u);
  checkSimpleBound(cstr.minus_l_leq_Ax_leq_minus_u);

  //with -Identity
  auto cstr2 = buildSimpleConstraints(-1, cstr.Ax_eq_0->variables()[0]);
  checkSimpleBound(cstr2.Ax_eq_0);
  checkSimpleBound(cstr2.Ax_geq_0);
  checkSimpleBound(cstr2.Ax_leq_0);
  checkSimpleBound(cstr2.Ax_eq_b);
  checkSimpleBound(cstr2.Ax_geq_b);
  checkSimpleBound(cstr2.Ax_leq_b);
  checkSimpleBound(cstr2.Ax_eq_minus_b);
  checkSimpleBound(cstr2.Ax_geq_minus_b);
  checkSimpleBound(cstr2.Ax_leq_minus_b);
  checkSimpleBound(cstr2.l_leq_Ax_leq_u);
  checkSimpleBound(cstr2.minus_l_leq_Ax_leq_minus_u);

  // with diagonal
  auto cstr3 = buildSimpleConstraints(2, cstr.Ax_eq_0->variables()[0]);
  checkSimpleBound(cstr3.Ax_eq_0);
  checkSimpleBound(cstr3.Ax_geq_0);
  checkSimpleBound(cstr3.Ax_leq_0);
  checkSimpleBound(cstr3.Ax_eq_b);
  checkSimpleBound(cstr3.Ax_geq_b);
  checkSimpleBound(cstr3.Ax_leq_b);
  checkSimpleBound(cstr3.Ax_eq_minus_b);
  checkSimpleBound(cstr3.Ax_geq_minus_b);
  checkSimpleBound(cstr3.Ax_leq_minus_b);
  checkSimpleBound(cstr3.l_leq_Ax_leq_u);
  checkSimpleBound(cstr3.minus_l_leq_Ax_leq_minus_u);

  // with diagonal (negative)
  auto cstr4 = buildSimpleConstraints(-2, cstr.Ax_eq_0->variables()[0]);
  checkSimpleBound(cstr4.Ax_eq_0);
  checkSimpleBound(cstr4.Ax_geq_0);
  checkSimpleBound(cstr4.Ax_leq_0);
  checkSimpleBound(cstr4.Ax_eq_b);
  checkSimpleBound(cstr4.Ax_geq_b);
  checkSimpleBound(cstr4.Ax_leq_b);
  checkSimpleBound(cstr4.Ax_eq_minus_b);
  checkSimpleBound(cstr4.Ax_geq_minus_b);
  checkSimpleBound(cstr4.Ax_leq_minus_b);
  checkSimpleBound(cstr4.l_leq_Ax_leq_u);
  checkSimpleBound(cstr4.minus_l_leq_Ax_leq_minus_u);

  std::vector<BLCPtr> c1 = {cstr.Ax_eq_0, cstr.Ax_geq_0, cstr.Ax_leq_0, cstr.Ax_eq_b,
                            cstr.Ax_geq_b, cstr.Ax_leq_b, cstr.Ax_eq_minus_b, cstr.Ax_geq_minus_b,
                            cstr.Ax_leq_minus_b, cstr.l_leq_Ax_leq_u, cstr.minus_l_leq_Ax_leq_minus_u };
  std::vector<BLCPtr> c2 = {cstr2.Ax_eq_0, cstr2.Ax_geq_0, cstr2.Ax_leq_0, cstr2.Ax_eq_b,
                            cstr2.Ax_geq_b, cstr2.Ax_leq_b, cstr2.Ax_eq_minus_b, cstr2.Ax_geq_minus_b,
                            cstr2.Ax_leq_minus_b, cstr2.l_leq_Ax_leq_u, cstr2.minus_l_leq_Ax_leq_minus_u };

  //check the intersection of bounds constraints
  for (size_t i = 0; i < 11; ++i)
  {
    for (size_t j = 0; j < 11; ++j)
    {
      checkSimpleBound(c1[i], c1[j]);
      checkSimpleBound(c1[i], c2[j]);
    }
  }
}

TEST_CASE("Test assignements with substitution")
{
  auto p0 = buildSimpleSubstitution(RHS::ZERO);
  auto p1 = buildSimpleSubstitution(RHS::AS_GIVEN);
  auto p2 = buildSimpleSubstitution(RHS::OPPOSITE);
  Substitution s0(p0.second, p0.second->variables()[1]); //substitution of y
  Substitution s1(p1.second, p1.second->variables()[1]); //substitution of y
  Substitution s2(p2.second, p2.second->variables()[1]); //substitution of y
  
  using T = std::vector<bool>;
  const bool t = true;
  const bool f = false;
  // constraint Ax = 0
  {
    //      Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v0 = { f,      f,      f,      t,      t,      t,      t,      t,      t,      f,      f };
    T v1 = { t,      f,      f,      t,      t,      t,      t,      t,      t,      f,      f };
    T v2 = { t,      f,      f,      t,      t,      t,      t,      t,      t,      f,      f };

    checkSimple(p0.first.Ax_eq_0, s0, v0);
    checkSimple(p1.first.Ax_eq_0, s1, v1);
    checkSimple(p2.first.Ax_eq_0, s2, v2);
  }

  // constraint Ax >= 0
  {
    //      Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v0 = { t,      t,      t,      f,      f,      f,      f,      f,      f,      f,      f };
    T v1 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    T v2 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };

    checkSimple(p0.first.Ax_geq_0, s0, v0);
    checkSimple(p1.first.Ax_geq_0, s1, v1);
    checkSimple(p2.first.Ax_geq_0, s2, v2);
  }

  // constraint Ax <= 0
  {
    //      Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v0 = { t,      t,      t,      f,      f,      f,      f,      f,      f,      f,      f };
    T v1 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    T v2 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };

    checkSimple(p0.first.Ax_leq_0, s0, v0);
    checkSimple(p1.first.Ax_leq_0, s1, v1);
    checkSimple(p2.first.Ax_leq_0, s2, v2);
  }

  // constraint Ax = 0
  {
    //      Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v0 = { t,      f,      f,      t,      t,      t,      t,      t,      t,      f,      f };
    T v1 = { t,      f,      f,      t,      t,      t,      t,      t,      t,      f,      f };
    T v2 = { t,      f,      f,      t,      t,      t,      t,      t,      t,      f,      f };

    checkSimple(p0.first.Ax_eq_b, s0, v0);
    checkSimple(p1.first.Ax_eq_b, s1, v1);
    checkSimple(p2.first.Ax_eq_b, s2, v2);
  }

  // constraint Ax >= b
  {
    //      Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v0 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    T v1 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    T v2 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };

    checkSimple(p0.first.Ax_geq_b, s0, v0);
    checkSimple(p1.first.Ax_geq_b, s1, v1);
    checkSimple(p2.first.Ax_geq_b, s2, v2);
  }

  // constraint Ax <= b
  {
    //      Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v0 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    T v1 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    T v2 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };

    checkSimple(p0.first.Ax_leq_b, s0, v0);
    checkSimple(p1.first.Ax_leq_b, s1, v1);
    checkSimple(p2.first.Ax_leq_b, s2, v2);
  }

  // constraint Ax >= b
  {
    //      Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v0 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    T v1 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    T v2 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };

    checkSimple(p0.first.l_leq_Ax_leq_u, s0, v0);
    checkSimple(p1.first.l_leq_Ax_leq_u, s1, v1);
    checkSimple(p2.first.l_leq_Ax_leq_u, s2, v2);
  }

  // constraint Ax <= b
  {
    //      Cx=0,  Cx=d,  Cx=-d,  Cx>=0,  Cx>=d, Cx>=-d,  Cx<=0,  Cx<=d, Cx<=-d,l<=Cx<=u,-l<=Cx<=-u
    T v0 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    T v1 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };
    T v2 = { t,      t,      t,      t,      f,      f,      t,      f,      f,      f,      f };

    checkSimple(p0.first.minus_l_leq_Ax_leq_minus_u, s0, v0);
    checkSimple(p1.first.minus_l_leq_Ax_leq_minus_u, s1, v1);
    checkSimple(p2.first.minus_l_leq_Ax_leq_minus_u, s2, v2);
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
