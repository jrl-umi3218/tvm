/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Space.h>
#include <tvm/Variable.h>
#include <tvm/constraint/BasicLinearConstraint.h>
#include <tvm/constraint/abstract/LinearConstraint.h>
#include <tvm/hint/Substitution.h>
#include <tvm/hint/internal/DiagonalCalculator.h>
#include <tvm/hint/internal/GenericCalculator.h>
#include <tvm/hint/internal/Substitutions.h>
#include <tvm/internal/MatrixProperties.h>
#include <tvm/internal/VariableVectorPartition.h>

#include <Eigen/SVD>

// #include <iostream>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm;
using namespace tvm::hint;
using namespace tvm::hint::internal;
using namespace Eigen;

TEST_CASE("GenericCalculator")
{
  VariablePtr x = Space(5).createVariable("x");
  VariablePtr y = Space(3).createVariable("y");
  MatrixXd A = MatrixXd::Random(5, 5);
  MatrixXd B = MatrixXd::Random(5, 3);
  VectorXd b = VectorXd::Random(5);

  std::shared_ptr<constraint::BasicLinearConstraint> c(
      new constraint::BasicLinearConstraint({A, B}, {x, y}, b, constraint::Type::EQUAL));

  auto calc = GenericCalculator().impl({std::static_pointer_cast<constraint::abstract::LinearConstraint>(c)}, {x}, 5);
  calc->update();

  MatrixXd AsA(5, 5);
  MatrixXd StA(0, 5);
  calc->premultiplyByASharpAndSTranspose(AsA, StA, A, false);
  FAST_CHECK_UNARY(AsA.isIdentity());

  MatrixXd AsB(5, 3);
  MatrixXd StB(0, 3);
  calc->premultiplyByASharpAndSTranspose(AsB, StB, B, true);
  auto qrA = A.colPivHouseholderQr();
  FAST_CHECK_UNARY(AsB.isApprox(-MatrixXd(qrA.solve(B))));

  MatrixXd C(3, 5);
  C << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
  MatrixXd D = MatrixXd::Random(3, 3);
  VectorXd d = VectorXd::Random(3);

  std::shared_ptr<constraint::BasicLinearConstraint> c2(
      new constraint::BasicLinearConstraint({C, D}, {x, y}, d, constraint::Type::EQUAL));

  auto calc2 = GenericCalculator().impl({std::static_pointer_cast<constraint::abstract::LinearConstraint>(c2)}, {x}, 2);
  calc2->update();
  MatrixXd CsD(5, 3);
  MatrixXd StD(1, 3);
  calc2->premultiplyByASharpAndSTranspose(CsD, StD, D, true);
  FAST_CHECK_UNARY(CsD.topRows(2).isApprox(-D.topRows(2)));
  FAST_CHECK_UNARY(CsD.bottomRows(3).isZero());
  FAST_CHECK_UNARY(StD.isApprox(D.row(2)));
  FAST_CHECK_UNARY(MatrixXd(C * calc2->N()).isZero());
  FAST_CHECK_EQ(calc2->N().colPivHouseholderQr().rank(), 3);
}

TEST_CASE("Diagonal Calculator")
{
  VariablePtr x = Space(7).createVariable("x");
  VariablePtr y = Space(3).createVariable("y");

  {
    MatrixXd A = MatrixXd::Identity(7, 7);
    MatrixXd B = MatrixXd::Random(7, 3);
    VectorXd b = VectorXd::Random(7);

    std::shared_ptr<constraint::BasicLinearConstraint> c(
        new constraint::BasicLinearConstraint({A, B}, {x, y}, b, constraint::Type::EQUAL));

    auto calc =
        DiagonalCalculator().impl({std::static_pointer_cast<constraint::abstract::LinearConstraint>(c)}, {x}, 7);
    calc->update();

    MatrixXd AsA(7, 7);
    MatrixXd StA(0, 7);
    calc->premultiplyByASharpAndSTranspose(AsA, StA, A, false);
    FAST_CHECK_UNARY(AsA.isIdentity());

    MatrixXd AsB(7, 3);
    MatrixXd StB(0, 3);
    calc->premultiplyByASharpAndSTranspose(AsB, StB, B, false);
    FAST_CHECK_UNARY(AsB.isApprox(B));

    FAST_CHECK_EQ(calc->N().rows(), 7);
    FAST_CHECK_EQ(calc->N().cols(), 0);
  }

  {
    MatrixXd A(3, 7);
    A << 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
    MatrixXd B = MatrixXd::Random(3, 3);
    VectorXd b = VectorXd::Random(3);

    std::shared_ptr<constraint::BasicLinearConstraint> c(
        new constraint::BasicLinearConstraint({A, B}, {x, y}, b, constraint::Type::EQUAL));

    auto calc = DiagonalCalculator({1, 3, 4}).impl(
        {std::static_pointer_cast<constraint::abstract::LinearConstraint>(c)}, {x}, 3);
    calc->update();

    MatrixXd AsA(7, 7);
    MatrixXd StA(0, 7);
    calc->premultiplyByASharpAndSTranspose(AsA, StA, A, false);
    FAST_CHECK_UNARY(AsA.isApprox(A.transpose() * A));

    MatrixXd AsB(7, 3);
    MatrixXd StB(0, 3);
    calc->premultiplyByASharpAndSTranspose(AsB, StB, B, false);
    FAST_CHECK_UNARY(AsB.isApprox(A.transpose() * B));

    MatrixXd N(7, 4);
    N << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    FAST_CHECK_UNARY(N.isApprox(calc->N()));
  }

  {
    MatrixXd A(5, 7);
    A << 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
    MatrixXd B = MatrixXd::Random(5, 3);
    VectorXd b = VectorXd::Random(5);

    std::shared_ptr<constraint::BasicLinearConstraint> c(
        new constraint::BasicLinearConstraint({A, B}, {x, y}, b, constraint::Type::EQUAL));

    auto calc = DiagonalCalculator({1, 3, 4}, {0, 2})
                    .impl({std::static_pointer_cast<constraint::abstract::LinearConstraint>(c)}, {x}, 3);
    calc->update();

    MatrixXd AsA(7, 7);
    MatrixXd StA(2, 7);
    calc->premultiplyByASharpAndSTranspose(AsA, StA, A, false);
    FAST_CHECK_UNARY(AsA.isApprox(A.transpose() * A));
    FAST_CHECK_UNARY(StA.isZero());

    MatrixXd AsB(7, 3);
    MatrixXd StB(2, 3);
    calc->premultiplyByASharpAndSTranspose(AsB, StB, B, false);
    FAST_CHECK_UNARY(AsB.isApprox(A.transpose() * B));
    FAST_CHECK_UNARY(StB.row(0).isApprox(B.row(0)));
    FAST_CHECK_UNARY(StB.row(1).isApprox(B.row(2)));

    MatrixXd N(7, 4);
    N << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    FAST_CHECK_UNARY(N.isApprox(calc->N()));
  }
}

MatrixXd randM(int m, int n, int r = 0)
{
  if(r <= 0 || r == std::min(m, n))
    return MatrixXd::Random(m, n);
  else
    return MatrixXd::Random(m, r) * MatrixXd::Random(r, n);
}

/** check whether the systems given by cstr and by subs are equivalent.
 * The system are supposed to be feasible.
 * From cstr, we deduce a system A [x;y] = b (1)
 * From subs, we deduce C [y;z] = d and x = E y + F z + g (2)
 * We check that any solution of (1) is a solution of (2) by writing a
 * solution of (1) [x;y] = pinv(A)*b + Na u with Na a base of the nullspace of
 * A and u a vector. For size(u) different value of u, we the rewrite (2) as a
 * system of z only, and verify we can find a solution.
 * To check that any solution of (2) yield a solution of (1), we first solve
 * C [y;z] = d to get [y;z] = pinv(C)*d + Nc v. Then for size(v) value of v, we
 * compute x from x = E y + F z + g, and check that [x;y] is a solution of (1).
 */
void checkEquivalence(const std::vector<std::shared_ptr<constraint::BasicLinearConstraint>> & cstr,
                      Substitutions & subs)
{
  subs.updateSubstitutions();

  VariableVector x(subs.variables());
  VariableVector y;
  VariableVector z(subs.additionalVariables());
  tvm::internal::VariableCountingVector partition(true);
  partition.add(x);
  for(const auto & c : cstr)
  {
    partition.add(c->variables());
  }
  int m0 = 0;
  for(auto & c : cstr)
  {
    m0 += c->size();
    for(const auto & vi : tvm::internal::VariableVectorPartition(c->variables(), partition))
    {
      if(!x.contains(*vi))
      {
        y.add(vi);
      }
    }
  }

  int m1 = 0;
  for(auto & c : subs.additionalConstraints())
  {
    m1 += c->size();
  }

  // Create A and b such that the system given by cstr is A [x;y] = b
  MatrixXd A = MatrixXd::Zero(m0, x.totalSize() + y.totalSize());
  VectorXd b(m0);
  m0 = 0;
  for(const auto & c : cstr)
  {
    auto mi = c->size();
    for(const auto & xi : x.variables())
    {
      if(c->variables().contains(*xi))
      {
        auto rx = xi->getMappingIn(x);
        A.block(m0, rx.start, mi, rx.dim) = c->jacobian(*xi);
      }
    }
    for(const auto & yi : y.variables())
    {
      if(c->variables().contains(*yi))
      {
        auto ry = yi->getMappingIn(y);
        A.block(m0, ry.start + x.totalSize(), mi, ry.dim) = c->jacobian(*yi);
      }
    }
    switch(c->rhs())
    {
      case constraint::RHS::ZERO:
        b.segment(m0, mi).setZero();
        break;
      case constraint::RHS::AS_GIVEN:
        b.segment(m0, mi) = c->e();
        break;
      case constraint::RHS::OPPOSITE:
        b.segment(m0, mi) = -c->e();
        break;
    }
    m0 += mi;
  }

  // Create C and d such that the additional constraint of subs write C [y;z] = d
  MatrixXd C = MatrixXd::Zero(m1, y.totalSize() + z.totalSize());
  VectorXd d(m1);
  m1 = 0;
  for(const auto & c : subs.additionalConstraints())
  {
    auto mi = c->size();
    for(const auto & yi : y.variables())
    {
      if(c->variables().contains(*yi))
      {
        auto ry = yi->getMappingIn(y);
        C.block(m1, ry.start, mi, ry.dim) = c->jacobian(*yi);
      }
    }
    for(const auto & zi : z.variables())
    {
      if(c->variables().contains(*zi))
      {
        auto rz = zi->getMappingIn(z);
        C.block(m1, rz.start + y.totalSize(), mi, rz.dim) = c->jacobian(*zi);
      }
    }
    switch(c->rhs())
    {
      case constraint::RHS::ZERO:
        d.segment(m1, mi).setZero();
        break;
      case constraint::RHS::AS_GIVEN:
        d.segment(m1, mi) = c->e();
        break;
      case constraint::RHS::OPPOSITE:
        d.segment(m1, mi) = -c->e();
        break;
    }
    m1 += mi;
  }

  // Create E, F and g such that the substitutions write x = E y + F z + g
  MatrixXd E = MatrixXd::Zero(x.totalSize(), y.totalSize());
  MatrixXd F = MatrixXd::Zero(x.totalSize(), z.totalSize());
  VectorXd g(x.totalSize());
  for(size_t i = 0; i < x.variables().size(); ++i)
  {
    const auto & f = subs.variableSubstitutions()[i];
    auto rx = x[static_cast<int>(i)]->getMappingIn(x);
    for(const auto & yi : y.variables())
    {
      if(f->variables().contains(*yi))
      {
        auto ry = yi->getMappingIn(y);
        E.block(rx.start, ry.start, rx.dim, ry.dim) = f->jacobian(*yi);
      }
    }
    for(const auto & zi : z.variables())
    {
      if(f->variables().contains(*zi))
      {
        auto rz = zi->getMappingIn(z);
        F.block(rx.start, rz.start, rx.dim, rz.dim) = f->jacobian(*zi);
      }
    }
    g.segment(rx.start, rx.dim) = f->b();
  }

  // Solve A [x;y] = b
  auto svdA = A.jacobiSvd(ComputeFullU | ComputeFullV);
  auto sol0 = svdA.solve(b);                                      // least square solution
  auto r0 = (A * sol0 - b).norm();                                // residual
  MatrixXd Na = svdA.matrixV().rightCols(A.cols() - svdA.rank()); // nullspace of A0
  FAST_CHECK_LE(r0, 1e-9);

  // Solve C [y;z] = d
  VectorXd sol1;
  MatrixXd Nc;
  if(C.size() > 0)
  {
    auto svdC = C.jacobiSvd(ComputeFullU | ComputeFullV);
    sol1 = svdC.solve(d);                                  // least square solution
    Nc = svdC.matrixV().rightCols(C.cols() - svdC.rank()); // nullspace of C
  }
  else
  {
    sol1 = VectorXd::Zero(C.cols());
    Nc = MatrixXd::Identity(C.cols(), C.cols());
  }

  // now check the equivalence:
  // 1 - The solutions of Solve C [y;z] = d give solutions of A [x;y] = b
  for(auto i = 0; i < Nc.cols(); ++i)
  {
    // one solution to the additional constraints
    VectorXd yz = sol1 + Nc * VectorXd::Random(Nc.cols());
    // split it over y and z
    y.set(yz.head(y.totalSize()));
    z.set(yz.tail(z.totalSize()));
    // compute the corresponding x
    subs.updateVariableValues();
    // compute the residual for the current x and y
    VectorXd xy(x.totalSize() + y.totalSize());
    xy.head(x.totalSize()) = x.value();
    xy.tail(y.totalSize()) = y.value();
    double res = (A * xy - b).norm();
    FAST_CHECK_LE(res, 1e-9);
  }

  // 2 - The solutions of A [x;y] = b are such that there we can find z for which
  //   C [y;z] = d and x = E y + F z + g
  MatrixXd M(C.rows() + F.rows(), z.totalSize());
  M.topRows(C.rows()) = C.rightCols(z.totalSize());
  M.bottomRows(F.rows()) = F;
  JacobiSVD<MatrixXd> svdM;
  if(z.totalSize() > 0)
  {
    svdM.compute(M, ComputeThinU | ComputeThinV);
  }
  for(auto i = 0; i < Na.cols(); ++i)
  {
    // one solution to the original system
    VectorXd xy = sol0 + Na * VectorXd::Random(Na.cols());
    // solve C2 z = d - C1 y and F z = x - E y - g
    VectorXd u(C.rows() + F.rows());
    u.head(C.rows()) = d - C.leftCols(y.totalSize()) * xy.tail(y.totalSize());
    u.tail(F.rows()) = xy.head(x.totalSize()) - E * xy.tail(y.totalSize()) - g;
    double res;
    if(z.totalSize() > 0)
    {
      VectorXd s = svdM.solve(u);
      res = (M * s - u).norm();
    }
    else
    {
      res = u.norm();
    }
    FAST_CHECK_LE(res, 1e-9);
  }
}

TEST_CASE("Substitution construction")
{
  using BLC = constraint::BasicLinearConstraint;
  auto eq = constraint::Type::EQUAL;

  VariablePtr x = Space(5).createVariable("x");
  {
    MatrixXd A = randM(5, 5);
    VectorXd b = VectorXd::Random(5);

    auto c = std::shared_ptr<BLC>(new BLC(A, x, b, eq));
    Substitution s(c, x);
    FAST_CHECK_EQ(s.rank(), 5);
    FAST_CHECK_EQ(typeid(*s.calculator()).hash_code(), typeid(GenericCalculator::Impl).hash_code());
  }

  {
    MatrixXd A = MatrixXd::Identity(5, 5);
    VectorXd b = VectorXd::Random(5);

    auto c = std::shared_ptr<BLC>(new BLC(5, x, eq));
    c->A(A, {tvm::internal::MatrixProperties::IDENTITY});
    Substitution s(c, x);
    FAST_CHECK_EQ(s.rank(), 5);
    FAST_CHECK_EQ(typeid(*s.calculator()).hash_code(), typeid(DiagonalCalculator::Impl).hash_code());
  }

  {
    MatrixXd A = randM(5, 5);
    VectorXd b = VectorXd::Random(5);

    auto c = std::shared_ptr<BLC>(new BLC(A, x, b, constraint::Type::LOWER_THAN));
    CHECK_THROWS(Substitution(c, x));

    VariablePtr y = Space(3).createVariable("y");
    CHECK_THROWS(Substitution(c, y));

    auto c2 = std::shared_ptr<BLC>(new BLC(3, y, eq));
    CHECK_THROWS(Substitution({c, c2}, x));

    auto c3 = std::shared_ptr<BLC>(new BLC(5, y, eq));
    CHECK_THROWS(Substitution(c3, y));
  }
}

TEST_CASE("Substitution0")
{
  using BLC = constraint::BasicLinearConstraint;
  auto eq = constraint::Type::EQUAL;

  {
    // solving A x = b using substitutions
    VariablePtr x = Space(5).createVariable("x");
    MatrixXd A = randM(5, 5);
    VectorXd b = VectorXd::Random(5);
    VectorXd x0 = A.colPivHouseholderQr().solve(b);

    auto c = std::shared_ptr<BLC>(new BLC(A, x, b, eq));
    Substitution s(c, x);
    Substitutions subs;
    subs.add(s);
    subs.finalize();
    FAST_CHECK_EQ(subs.variables().size(), 1);
    FAST_CHECK_EQ(subs.variables().front(), x);
    FAST_CHECK_EQ(subs.additionalVariables().size(), 0);
    FAST_CHECK_EQ(subs.additionalConstraints().size(), 0);
    FAST_CHECK_EQ(subs.variableSubstitutions().size(), 1);

    auto f = subs.variableSubstitutions().front();
    FAST_CHECK_EQ(f->variables().totalSize(), 0);
    subs.updateSubstitutions();
    FAST_CHECK_UNARY(f->b().isApprox(x0));
    subs.updateVariableValues();
    FAST_CHECK_UNARY(x->value().isApprox(x0));
  }
}

TEST_CASE("Substitution1")
{
  int m1 = 3;
  int n1 = 4;
  int r1 = 2;
  int m2 = 3;
  int n2 = 3;
  int r2 = 3;
  int m3 = 3;
  int n3 = 6;
  int r3 = 3;
  int m4 = 4;
  int n4 = 4;
  int r4 = 4;
  int m5 = 7;
  int n5 = 8;
  int r5 = 4;
  int m6 = 3;
  int n6 = 3;
  int r6 = 3;
  int l1 = 3;
  int l2 = 7;
  int l3 = 4;
  int l4 = 4;

  VariablePtr x1 = Space(n1).createVariable("x1");
  VariablePtr x2 = Space(n2).createVariable("x2");
  VariablePtr x3 = Space(n3).createVariable("x3");
  VariablePtr x4 = Space(n4).createVariable("x4");
  VariablePtr x5 = Space(n5).createVariable("x5");
  VariablePtr x6 = Space(n6).createVariable("x6");
  VariablePtr y1 = Space(l1).createVariable("y1");
  VariablePtr y2 = Space(l2).createVariable("y2");
  VariablePtr y3 = Space(l3).createVariable("y3");
  VariablePtr y4 = Space(l4).createVariable("y4");

  VectorXd b1 = VectorXd::Random(m1);
  VectorXd b2 = VectorXd::Random(m2);
  VectorXd b3 = VectorXd::Random(m3);
  VectorXd b4 = VectorXd::Random(m4);
  VectorXd b5 = VectorXd::Random(m5);
  VectorXd b6 = VectorXd::Random(m6);

  //                                             | x1 |
  //                                             | x2 |
  // | A11 A12 A13  0   0  A16 B11  0   0   0  | | x3 |   | b1 |
  // |  0  A22  0  A24  0   0   0   0   0   0  | | x4 |   | b2 |
  // |  0   0  A33  0  A35 A36 B31  0   0   0  | | x5 |   | b3 |
  // |  0   0   0  A44 A45  0   0   0   0   0  | | x6 | = | b4 |
  // |  0   0   0   0  A55  0   0  B52  0  B54 | | y1 |   | b5 |
  // |  0   0   0   0   0  A66  0  B62 B63  0  | | y2 |   | b6 |
  //                                             | y3 |
  //                                             | y4 |

  MatrixXd A11 = randM(m1, n1, r1);
  MatrixXd A12 = randM(m1, n2);
  MatrixXd A13 = randM(m1, n3);
  MatrixXd A16 = randM(m1, n6);
  MatrixXd B11 = randM(m1, l1);
  MatrixXd A22 = randM(m2, n2, r2);
  MatrixXd A24 = randM(m2, n4);
  MatrixXd A33 = randM(m3, n3, r3);
  MatrixXd A35 = randM(m3, n5);
  MatrixXd A36 = randM(m3, n6);
  MatrixXd B31 = randM(m3, l1);
  MatrixXd A44 = randM(m4, n4, r4);
  MatrixXd A45 = randM(m4, n5);
  MatrixXd A55 = randM(m5, n5, r5);
  MatrixXd B52 = randM(m5, l2);
  MatrixXd B54 = randM(m5, l4);
  MatrixXd A66 = randM(m6, n6, r6);
  MatrixXd B62 = randM(m6, l2);
  MatrixXd B63 = randM(m6, l3);

  using BLC = constraint::BasicLinearConstraint;
  auto eq = constraint::Type::EQUAL;
  auto c1 = std::shared_ptr<BLC>(new BLC({A11, A12, A13, A16, B11}, {x1, x2, x3, x6, y1}, b1, eq));
  auto c2 = std::shared_ptr<BLC>(new BLC({A22, A24}, {x2, x4}, b2, eq));
  auto c3 = std::shared_ptr<BLC>(new BLC({A33, A35, A36, B31}, {x3, x5, x6, y1}, b3, eq));
  auto c4 = std::shared_ptr<BLC>(new BLC({A44, A45}, {x4, x5}, b4, eq));
  auto c5 = std::shared_ptr<BLC>(new BLC({A55, B52, B54}, {x5, y2, y4}, b5, eq));
  auto c6 = std::shared_ptr<BLC>(new BLC({A66, B62, B63}, {x6, y2, y3}, b6, eq));

  Substitution s1(c1, x1, r1);
  Substitution s2(c2, x2, r2);
  Substitution s3(c3, x3, r3);
  Substitution s4(c4, x4, r4);
  Substitution s5(c5, x5, r5);
  Substitution s6(c6, x6, r6);

  Substitutions subs;
  subs.add(s1);
  subs.add(s2);
  subs.add(s3);
  subs.add(s4);
  subs.add(s5);
  subs.add(s6);

  subs.finalize();

  checkEquivalence({c1, c2, c3, c4, c5, c6}, subs);
}

TEST_CASE("Substitution2")
{
  int m1 = 3;
  int n1 = 4;
  int m2 = 3;
  int n2 = 3;
  int m3 = 3;
  int n3 = 6;
  int m4 = 5;
  int n4 = 4;
  int m5 = 7;
  int n5 = 5;
  int m6 = 3;
  int n6 = 2;
  int m7 = 6;
  int n7 = 7;
  int m8 = 4;
  int n8 = 4;
  int m9 = 4;
  int n9 = 3;
  int l1 = 3;
  int l2 = 7;

  VariablePtr x1 = Space(n1).createVariable("x1");
  VariablePtr x2 = Space(n2).createVariable("x2");
  VariablePtr x3 = Space(n3).createVariable("x3");
  VariablePtr x4 = Space(n4).createVariable("x4");
  VariablePtr x5 = Space(n5).createVariable("x5");
  VariablePtr x6 = Space(n6).createVariable("x6");
  VariablePtr x7 = Space(n7).createVariable("x7");
  VariablePtr x8 = Space(n8).createVariable("x8");
  VariablePtr x9 = Space(n9).createVariable("x9");
  VariablePtr y1 = Space(l1).createVariable("y1");
  VariablePtr y2 = Space(l2).createVariable("y2");

  //                                                 | x1 |
  // |  0   0   0  A14  0   0  A17  0  A19 B11  0  | | x2 |   | b1 |
  // |  0  A22  0   0   0   0   0   0   0  B21 B22 | | x3 |   | b2 |
  // |  0   0   0   0   0  A36  0  A38  0  B31  0  | | x4 |   | b3 |
  // |  0   0   0   0  A45  0   0   0   0  B41  0  | | x5 |   | b4 |
  // |  0   0   0  A54  0   0  A57  0  A59  0   0  | | x6 | = | b5 |
  // |  0   0   0   0   0  A66  0  A68  0  B61 B62 | | x7 |   | b6 |
  // |  0   0  A73  0  A75  0   0   0   0   0  B72 | | x8 |   | b7 |
  // | A81  0   0   0   0   0  A87  0   0  B81  0  | | x9 |   | b8 |
  // |  0   0   0  A94  0   0   0   0   0   0  B92 | | y1 |   | b9 |
  //                                                 | y2 |

  MatrixXd A14 = randM(m1, n4);
  MatrixXd A17 = randM(m1, n7);
  MatrixXd A19 = randM(m1, n9);
  MatrixXd B11 = randM(m1, l1);
  MatrixXd A22 = randM(m2, n2);
  MatrixXd B21 = randM(m2, l1);
  MatrixXd B22 = randM(m2, l2);
  MatrixXd A36 = randM(m3, n6);
  MatrixXd A38 = randM(m3, n8);
  MatrixXd B31 = randM(m3, l1);
  MatrixXd A45 = randM(m4, n5);
  MatrixXd B41 = randM(m4, l1);
  MatrixXd A54 = randM(m5, n4);
  MatrixXd A57 = randM(m5, n7);
  MatrixXd A59 = randM(m5, n9);
  MatrixXd A66 = randM(m6, n6);
  MatrixXd A68 = randM(m6, n8);
  MatrixXd B61 = randM(m6, l1);
  MatrixXd B62 = randM(m6, l2);
  MatrixXd A73 = randM(m7, n3);
  MatrixXd A75 = randM(m7, n5);
  MatrixXd B72 = randM(m7, l2);
  MatrixXd A81 = randM(m8, n1);
  MatrixXd A87 = randM(m8, n7);
  MatrixXd B81 = randM(m8, l1);
  MatrixXd A94 = randM(m9, n4);
  MatrixXd B92 = randM(m9, l2);
  VectorXd b1 = VectorXd::Random(m1);
  VectorXd b2 = VectorXd::Random(m2);
  VectorXd b3 = VectorXd::Random(m3);
  VectorXd b4 = VectorXd::Random(m4);
  VectorXd b5 = VectorXd::Random(m5);
  VectorXd b6 = VectorXd::Random(m6);
  VectorXd b7 = VectorXd::Random(m7);
  VectorXd b8 = VectorXd::Random(m8);
  VectorXd b9 = VectorXd::Random(m9);

  using BLC = constraint::BasicLinearConstraint;
  auto eq = constraint::Type::EQUAL;
  auto c1 = std::shared_ptr<BLC>(new BLC({A14, A17, A19, B11}, {x4, x7, x9, y1}, b1, eq));
  auto c2 = std::shared_ptr<BLC>(new BLC({A22, B21, B22}, {x2, y1, y2}, b2, eq));
  auto c3 = std::shared_ptr<BLC>(new BLC({A36, A38, B31}, {x6, x8, y1}, b3, eq));
  auto c4 = std::shared_ptr<BLC>(new BLC({A45, B41}, {x5, y1}, b4, eq));
  auto c5 = std::shared_ptr<BLC>(new BLC({A54, A57, A59}, {x4, x7, x9}, b5, eq));
  auto c6 = std::shared_ptr<BLC>(new BLC({A66, A68, B62, B61}, {x6, x8, y2, y1}, b6, eq));
  auto c7 = std::shared_ptr<BLC>(new BLC({A73, A75, B72}, {x3, x5, y2}, b7, eq));
  auto c8 = std::shared_ptr<BLC>(new BLC({A81, A87, B81}, {x1, x7, y1}, b8, eq));
  auto c9 = std::shared_ptr<BLC>(new BLC({A94, B92}, {x4, y2}, b9, eq));

  Substitution s1(c1, x9);
  Substitution s2(c2, x2);
  Substitution s3(std::vector<LinearConstraintPtr>{c3, c6}, std::vector<VariablePtr>{x6, x8});
  Substitution s4(c4, x5);
  Substitution s5(c5, x7);
  // no s6: c3 and c6 are merged
  Substitution s7(c7, x3);
  Substitution s8(c8, x1);
  Substitution s9(c9, x4);

  Substitutions subs;
  subs.add(s1);
  subs.add(s2);
  subs.add(s3);
  subs.add(s4);
  subs.add(s5);
  subs.add(s7);
  subs.add(s8);
  subs.add(s9);

  subs.finalize();
  checkEquivalence({c1, c2, c3, c4, c5, c6, c7, c8, c9}, subs);
}

/** Generate a random number r, min <= r < max*/
int randI(int min, int max)
{
  assert(min < max);
  return (rand() % (max - min)) + min;
}

/** Return true with a probability p*/
bool randP(double p)
{
  assert(0 <= p && p <= 1);
  return static_cast<double>(rand()) / RAND_MAX < p;
}

/** Create a random system A [x;y] = b*/
void randomSubstitutions()
{
  using BLC = constraint::BasicLinearConstraint;
  auto eq = constraint::Type::EQUAL;

  int nxmin = 4;
  int nxmax = 16;
  int nymin = 0;
  int nymax = 9;
  int nmin = 2;
  int nmax = 16;
  double nonFullRankP = 0.33;
  int nx = randI(nxmin, nxmax); // number of x variables
  int ny = randI(nymin, nymax); // number of y variables
  double px = 0.4;              // probability to have a non-zero off-diagonal block in the 'x part' of A
  double py = 0.3;              // probability to have a non-zero block in the 'y part' of B

  std::vector<VariablePtr> x;
  std::vector<VariablePtr> y;
  std::vector<int> m;
  std::vector<int> n;
  std::vector<int> r;
  std::vector<int> l;
  std::vector<std::shared_ptr<BLC>> cstr;

  bool fullRankSystem = false;

  // We restart from scratch when we do not get a system that is full rank
  while(!fullRankSystem)
  {
    int ma = 0;
    int na = 0;
    x.clear();
    y.clear();
    m.clear();
    n.clear();
    r.clear();
    l.clear();
    cstr.clear();
    for(int i = 0; i < nx; ++i)
    {
      std::stringstream ss;
      ss << "x" << i;
      int ni = randI(nmin, nmax);
      int mi = 1 + randI(1, ni);
      n.push_back(ni);
      m.push_back(mi);
      x.push_back(Space(ni).createVariable(ss.str()));
      if(randP(nonFullRankP))
      {
        r.push_back(randI(mi / 2, mi + 1));
      }
      else
      {
        r.push_back(mi);
      }
      ma += mi;
      na += ni;
    }
    for(int i = 0; i < ny; ++i)
    {
      std::stringstream ss;
      ss << "y" << i;
      int ni = randI(nmin, nmax);
      l.push_back(ni);
      y.push_back(Space(ni).createVariable(ss.str()));
      na += l.back();
    }

    for(size_t i = 0; i < static_cast<size_t>(nx); ++i)
    {
      std::vector<VariablePtr> v;
      std::vector<MatrixXd> M;
      std::vector<MatrixConstRef> Mr;
      for(size_t j = 0; j < static_cast<size_t>(nx); ++j)
      {
        if(i == j)
        {
          M.push_back(randM(m[i], n[j], r[i]));
          v.push_back(x[j]);
        }
        else
        {
          if(randP(px))
          {
            M.push_back(randM(m[i], n[j]));
            v.push_back(x[j]);
          }
        }
      }
      for(size_t j = 0; j < static_cast<size_t>(ny); ++j)
      {
        if(randP(py))
        {
          M.push_back(randM(m[i], l[j]));
          v.push_back(y[j]);
        }
      }
      for(const auto & Mi : M)
      {
        Mr.push_back(Mi);
      }
      VectorXd b = VectorXd::Random(m[i]);
      auto c = std::shared_ptr<BLC>(new BLC(Mr, v, b, eq));
      cstr.push_back(c);
    }
    // check if the system is feasible
    MatrixXd A = MatrixXd::Zero(ma, na);
    int ms = 0;
    for(const auto & c : cstr)
    {
      int ns = 0;
      int mi = c->size();
      for(const auto & xi : x)
      {
        int ni = xi->size();
        if(c->variables().contains(*xi))
        {
          A.block(ms, ns, mi, ni) = c->jacobian(*xi);
        }
        ns += ni;
      }
      for(const auto & yi : y)
      {
        int ni = yi->size();
        if(c->variables().contains(*yi))
        {
          A.block(ms, ns, mi, ni) = c->jacobian(*yi);
        }
        ns += ni;
      }
      ms += mi;
    }
    auto svd = A.jacobiSvd();
    fullRankSystem = svd.rank() == A.rows();
  }

  Substitutions subs;
  for(size_t i = 0; i < cstr.size(); ++i)
  {
    Substitution s(cstr[i], x[i], r[i]);
    subs.add(s);
  }
  subs.finalize();
  checkEquivalence(cstr, subs);
}

TEST_CASE("Random substitutions")
{
  // In some very rare instance, this test can fail because of rank issues
  // during the qr decomposition of GenericCalculator. This is because of the
  // heuristic taken to get the rank of a group of constraints. This is not to
  // be regarded as an issue.
  for(int i = 0; i < 10; ++i)
  {
    randomSubstitutions();
  }
}

TEST_CASE("Substitution with subvariables")
{
  VariablePtr x = Space(8).createVariable("x");
  VariablePtr x1 = x->subvariable(3, "x1", 0);
  VariablePtr x2 = x->subvariable(5, "x2", 3);

  MatrixXd M = randM(8, 8);
  VectorXd r = VectorXd::Random(8);

  auto A1 = M.topRows(5);
  auto A2 = M.bottomRows(3);
  auto b1 = r.head(5);
  auto b2 = r.tail(3);

  using BLC = constraint::BasicLinearConstraint;
  auto eq = constraint::Type::EQUAL;
  VectorXd x0 = M.colPivHouseholderQr().solve(r);

  auto c = std::shared_ptr<BLC>(new BLC(A1, x, b1, eq));
  Substitution s(c, x2);
  Substitutions subs;
  subs.add(s);
  subs.finalize();
  FAST_CHECK_EQ(subs.variables().size(), 1);
  FAST_CHECK_EQ(*subs.variables().front(), *x2);
  FAST_CHECK_EQ(subs.additionalVariables().size(), 0);
  FAST_CHECK_EQ(subs.otherVariables().size(), 1);
  FAST_CHECK_EQ(*subs.otherVariables().front(), *x1);
  FAST_CHECK_EQ(subs.additionalConstraints().size(), 0);
  FAST_CHECK_EQ(subs.variableSubstitutions().size(), 1);

  auto f = subs.variableSubstitutions().front();
  FAST_CHECK_EQ(f->variables().numberOfVariables(), 1);
  FAST_CHECK_EQ(*f->variables()[0], *x1);
  subs.updateSubstitutions();

  MatrixXd C = A2.leftCols(3) + A2.rightCols(5) * f->jacobian(*x1);
  VectorXd d = b2 - A2.rightCols(5) * f->b();
  VectorXd y = C.colPivHouseholderQr().solve(d);
  x1 << y;

  subs.updateVariableValues();
  FAST_CHECK_UNARY(x->value().isApprox(x0));
}
