/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Range.h>
#include <tvm/Variable.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm;

TEST_CASE("Space product")
{
  Space R2(2);
  Space R3(3);
  Space SO3(Space::Type::SO3);
  Space S2(2, 3, 3);

  Space R5 = R2 * R3;
  FAST_CHECK_EQ(R5.size(), 5);
  FAST_CHECK_EQ(R5.rSize(), 5);
  FAST_CHECK_EQ(R5.tSize(), 5);
  FAST_CHECK_EQ(R5.type(), Space::Type::Euclidean);

  Space SE3 = R3 * SO3;
  FAST_CHECK_EQ(SE3.size(), 6);
  FAST_CHECK_EQ(SE3.rSize(), 7);
  FAST_CHECK_EQ(SE3.tSize(), 6);
  FAST_CHECK_EQ(SE3.type(), Space::Type::Unspecified);

  Space S = R2 * S2;
  FAST_CHECK_EQ(S.size(), 4);
  FAST_CHECK_EQ(S.rSize(), 5);
  FAST_CHECK_EQ(S.tSize(), 5);
  FAST_CHECK_EQ(S.type(), Space::Type::Unspecified);
}

TEST_CASE("Test Variable creation")
{
  VariablePtr u = Space(3).createVariable("u");
  FAST_CHECK_EQ(u->size(), 3);
  FAST_CHECK_EQ(dot(u)->size(), 3);
  FAST_CHECK_EQ(dot(u, 4)->size(), 3);
  FAST_CHECK_UNARY(u->space().isEuclidean());
  FAST_CHECK_EQ(u->space().size(), 3);
  FAST_CHECK_EQ(u->space().rSize(), 3);
  FAST_CHECK_EQ(u->space().tSize(), 3);
  FAST_CHECK_UNARY(u->isEuclidean());
  FAST_CHECK_UNARY(dot(u)->isEuclidean());
  FAST_CHECK_EQ(u->space().type(), Space::Type::Euclidean);

  VariablePtr v = Space(3, 4, 3).createVariable("v");
  FAST_CHECK_EQ(v->size(), 4);
  FAST_CHECK_EQ(dot(v)->size(), 3);
  FAST_CHECK_EQ(dot(v, 4)->size(), 3);
  FAST_CHECK_UNARY(!v->space().isEuclidean());
  FAST_CHECK_EQ(v->space().size(), 3);
  FAST_CHECK_EQ(v->space().rSize(), 4);
  FAST_CHECK_EQ(v->space().tSize(), 3);
  FAST_CHECK_UNARY_FALSE(v->isEuclidean());
  FAST_CHECK_UNARY(dot(v)->isEuclidean());
  FAST_CHECK_EQ(v->space().type(), Space::Type::Unspecified);

  VariablePtr w = v->duplicate("w");
  FAST_CHECK_NE(v, w);
  FAST_CHECK_EQ(v->space(), w->space());
  FAST_CHECK_UNARY_FALSE(w->isEuclidean());
  FAST_CHECK_UNARY(dot(w)->isEuclidean());

  VariablePtr x = Space(Space::Type::SO3).createVariable("x");
  FAST_CHECK_EQ(x->size(), 4);
  FAST_CHECK_EQ(dot(x)->size(), 3);
  FAST_CHECK_UNARY_FALSE(x->space().isEuclidean());
  FAST_CHECK_EQ(x->space().type(), Space::Type::SO3);

  VariablePtr y = Space(Space::Type::SE3).createVariable("y");
  FAST_CHECK_EQ(y->size(), 7);
  FAST_CHECK_EQ(dot(y)->size(), 6);
  FAST_CHECK_UNARY_FALSE(y->space().isEuclidean());
  FAST_CHECK_EQ(y->space().type(), Space::Type::SE3);
}

TEST_CASE("Test Variable value")
{
  {
    VariablePtr v = Space(3).createVariable("v");
    Eigen::Vector3d val = Eigen::Vector3d::Random();
    v->value(val);

    FAST_CHECK_UNARY(v->value().isApprox(val));
    CHECK_THROWS(v->value(Eigen::VectorXd(5)));
  }
  {
    VariablePtr v = Space(3).createVariable("v");
    Eigen::Vector3d val(1, 2, 3);
    v << val;
    FAST_CHECK_UNARY(v->value().isApprox(val));
  }
  {
    VariablePtr v = Space(3).createVariable("v");
    Eigen::VectorXd val(5);
    val << 1, 2, 3, 4, 5;
    v << val.head(3);
    FAST_CHECK_UNARY(v->value().isApprox(Eigen::Vector3d(1, 2, 3)));
  }
  {
    VariablePtr v = Space(3).createVariable("v");
    v << 1, 2, 3;
    FAST_CHECK_UNARY(v->value().isApprox(Eigen::Vector3d(1, 2, 3)));
  }
  {
    VariablePtr v = Space(3).createVariable("v");
    Eigen::VectorXd val(5);
    val << 1, 2, 3, 4, 5;
    v << val.tail(2), 6;
    FAST_CHECK_UNARY(v->value().isApprox(Eigen::Vector3d(4, 5, 6)));
  }
}

TEST_CASE("Test Variable Derivatives")
{
  VariablePtr v = Space(3).createVariable("v");
  FAST_CHECK_EQ(v->derivativeNumber(), 0);
  FAST_CHECK_UNARY(v->isBasePrimitive());
  FAST_CHECK_EQ(v->basePrimitive(), v);
  CHECK_THROWS(v->primitive());

  auto dv = dot(v);
  FAST_CHECK_EQ(dv->space(), v->space());
  FAST_CHECK_EQ(dv->derivativeNumber(), 1);
  FAST_CHECK_UNARY(!dv->isBasePrimitive());
  FAST_CHECK_EQ(dv->basePrimitive(), v);
  FAST_CHECK_EQ(dv->primitive(), v);
  CHECK_THROWS(dv->primitive<3>());

  auto dv3 = dot(v, 3);
  FAST_CHECK_EQ(dv3->space(), v->space());
  FAST_CHECK_EQ(dv3->derivativeNumber(), 3);
  FAST_CHECK_UNARY(!dv3->isBasePrimitive());
  FAST_CHECK_EQ(dv3->primitive<3>(), v);
  FAST_CHECK_EQ(dv3->basePrimitive(), v);
  FAST_CHECK_EQ(dot(dv, 2), dv3);
  FAST_CHECK_EQ(dot(dv), dv3->primitive());

  VariablePtr u = Space(4).createVariable("u");
  FAST_CHECK_UNARY(dv->isDerivativeOf(*v));
  FAST_CHECK_UNARY_FALSE(v->isDerivativeOf(*dv));
  FAST_CHECK_UNARY_FALSE(dv->isDerivativeOf(*u));
  FAST_CHECK_UNARY_FALSE(dv->isDerivativeOf(*dv));
  FAST_CHECK_UNARY(v->isPrimitiveOf(*dv));
  FAST_CHECK_UNARY_FALSE(dv->isPrimitiveOf(*v));
  FAST_CHECK_UNARY_FALSE(dv->isPrimitiveOf(*u));
  FAST_CHECK_UNARY_FALSE(dv->isPrimitiveOf(*dv));
  FAST_CHECK_UNARY(dv3->isDerivativeOf(*v));
  FAST_CHECK_UNARY(dv3->isDerivativeOf(*dv));
}

TEST_CASE("Test Variable Name")
{
  VariablePtr v = Space(3).createVariable("v");
  FAST_CHECK_EQ(v->name(), "v");
  auto dv = dot(v);
  FAST_CHECK_EQ(dv->name(), "d v / dt");
  auto dv3 = dot(dv, 2);
  FAST_CHECK_EQ(dv3->name(), "d3 v / dt3");
  auto dv5 = dot(v, 5);
  FAST_CHECK_EQ(dv5->name(), "d5 v / dt5");
  auto dv4 = dot(dv, 3);
  FAST_CHECK_EQ(dv4->name(), "d4 v / dt4");
  auto du3 = dv3->duplicate();
  FAST_CHECK_EQ(du3->name(), "d3 v' / dt3");
  auto dw3 = dv3->duplicate("w");
  FAST_CHECK_EQ(dw3->name(), "d3 w / dt3");
}

TEST_CASE("Test derivative lifetime")
{
  VariablePtr x = Space(3).createVariable("x");
  {
    dot(x)->value(Eigen::Vector3d(3, 1, 4));
  }
  FAST_CHECK_EQ(dot(x)->value()[0], 3);
  FAST_CHECK_EQ(dot(x)->value()[1], 1);
  FAST_CHECK_EQ(dot(x)->value()[2], 4);
}

VariablePtr createDerivative(const std::string & name, int ndiff)
{
  VariablePtr x = Space(3).createVariable(name);
  return dot(x, ndiff);
}

std::weak_ptr<Variable> testRefCounting()
{
  // Creating a variable and its derivative, but loosing the ptr on the variable
  VariablePtr dx = createDerivative("x", 1);
  FAST_CHECK_EQ(dx.use_count(), 1);

  // Getting back the primitive, checking the ref count
  {
    VariablePtr x = dx->primitive();
    FAST_CHECK_EQ(dx.use_count(), 2);
    FAST_CHECK_EQ(x.use_count(), 2);
  }
  FAST_CHECK_EQ(dx.use_count(), 1);

  // Creating a derivative, but loosing the shared_ptr on it
  Variable *pddxa, *pddxb;
  {
    VariablePtr ddx = dot(dx);
    pddxa = ddx.get();
  }
  // Getting the derivative again, with another way
  {
    VariablePtr d3x = dot(dx, 2);
    pddxb = d3x->primitive().get();
  }
  // Checking the derivative was both time the same
  FAST_CHECK_EQ(pddxa, pddxb);

  return dx;
}

TEST_CASE("Test derivatives ref counting")
{
  auto dx = testRefCounting();
  // Checking dx was destroyed.
  FAST_CHECK_UNARY(dx.expired());
}

TEST_CASE("Subvariable")
{
  // Basic creation and derivation
  VariablePtr v = Space(6, 7, 6).createVariable("v");
  VariablePtr v2 = v->subvariable(Space(3), "v2", Space(3, 4, 3));
  FAST_CHECK_EQ(v2->size(), 3);
  FAST_CHECK_EQ(dot(v2)->size(), 3);
  FAST_CHECK_EQ(dot(v2, 4)->size(), 3);
  FAST_CHECK_UNARY(v2->space().isEuclidean());
  FAST_CHECK_EQ(v2->space().size(), 3);
  FAST_CHECK_EQ(v2->space().rSize(), 3);
  FAST_CHECK_EQ(v2->space().tSize(), 3);
  FAST_CHECK_UNARY(v2->isEuclidean());
  FAST_CHECK_UNARY(dot(v2)->isEuclidean());
  FAST_CHECK_EQ(v2->space().type(), Space::Type::Euclidean);
  v2 << Eigen::VectorXd::Ones(3);
  FAST_CHECK_UNARY(v->value().isApprox((Eigen::VectorXd(7) << 0, 0, 0, 0, 1, 1, 1).finished()));

  // Duplicate
  VariablePtr w2 = v2->duplicate();
  FAST_CHECK_NE(*v2, *w2);
  FAST_CHECK_EQ(w2->size(), 3);
  FAST_CHECK_EQ(dot(w2)->size(), 3);

  // Subvariable of subvariable
  VariablePtr v21 = v2->subvariable(Space(2), "v21");
  FAST_CHECK_EQ(v21->size(), 2);
  FAST_CHECK_EQ(dot(v21)->size(), 2);
  FAST_CHECK_EQ(dot(v21, 4)->size(), 2);
  FAST_CHECK_UNARY(v21->space().isEuclidean());
  FAST_CHECK_EQ(v21->space().size(), 2);
  FAST_CHECK_EQ(v21->space().rSize(), 2);
  FAST_CHECK_EQ(v21->space().tSize(), 2);
  FAST_CHECK_UNARY(v21->isEuclidean());
  FAST_CHECK_UNARY(dot(v21)->isEuclidean());
  FAST_CHECK_EQ(v21->space().type(), Space::Type::Euclidean);
  v21 << Eigen::VectorXd::Constant(2, 3);
  FAST_CHECK_UNARY(v->value().isApprox((Eigen::VectorXd(7) << 0, 0, 0, 0, 3, 3, 1).finished()));

  // Subvariable of derivative
  VariablePtr d2v2 = dot(v, 2)->subvariable(Space(3), "d2v2", Space(3, 4, 3));
  FAST_CHECK_EQ(d2v2->size(), 3);
  FAST_CHECK_EQ(dot(d2v2)->size(), 3);
  FAST_CHECK_EQ(dot(d2v2, 4)->size(), 3);
  FAST_CHECK_UNARY(d2v2->space().isEuclidean());
  FAST_CHECK_EQ(d2v2->space().size(), 3);
  FAST_CHECK_EQ(d2v2->space().rSize(), 3);
  FAST_CHECK_EQ(d2v2->space().tSize(), 3);
  FAST_CHECK_UNARY(d2v2->isEuclidean());
  FAST_CHECK_UNARY(dot(d2v2)->isEuclidean());
  FAST_CHECK_EQ(d2v2->space().type(), Space::Type::Euclidean);
  d2v2 << Eigen::VectorXd::Ones(3);
  FAST_CHECK_UNARY(dot(v, 2)->value().isApprox((Eigen::VectorXd(6) << 0, 0, 0, 1, 1, 1).finished()));
}

TEST_CASE("Equality")
{
  Space S(10);
  VariablePtr u = S.createVariable("u");
  VariablePtr v = S.createVariable("v");
  VariablePtr u1 = u->subvariable(Space(5), "u1", Space(2));
  VariablePtr u2 = u->subvariable(Space(7), "u2");
  VariablePtr u3 = u->subvariable(Space(10), "u3");
  VariablePtr u21 = u2->subvariable(Space(5), "u21", Space(2));

  FAST_CHECK_EQ(*u, *u);
  FAST_CHECK_NE(*u, *v);
  FAST_CHECK_NE(*u, *u1);
  FAST_CHECK_NE(*u, *u2);
  FAST_CHECK_EQ(*u, *u3);
  FAST_CHECK_NE(*u, *u21);

  FAST_CHECK_NE(*v, *u);
  FAST_CHECK_EQ(*v, *v);
  FAST_CHECK_NE(*v, *u1);
  FAST_CHECK_NE(*v, *u2);
  FAST_CHECK_NE(*v, *u3);
  FAST_CHECK_NE(*v, *u21);

  FAST_CHECK_NE(*u1, *u);
  FAST_CHECK_NE(*u1, *v);
  FAST_CHECK_EQ(*u1, *u1);
  FAST_CHECK_NE(*u1, *u2);
  FAST_CHECK_NE(*u1, *u3);
  FAST_CHECK_EQ(*u1, *u21);

  FAST_CHECK_NE(*u2, *u);
  FAST_CHECK_NE(*u2, *v);
  FAST_CHECK_NE(*u2, *u1);
  FAST_CHECK_EQ(*u2, *u2);
  FAST_CHECK_NE(*u2, *u3);
  FAST_CHECK_NE(*u2, *u21);

  FAST_CHECK_EQ(*u3, *u);
  FAST_CHECK_NE(*u3, *v);
  FAST_CHECK_NE(*u3, *u1);
  FAST_CHECK_NE(*u3, *u2);
  FAST_CHECK_EQ(*u3, *u3);
  FAST_CHECK_NE(*u3, *u21);

  FAST_CHECK_NE(*u21, *u);
  FAST_CHECK_NE(*u21, *v);
  FAST_CHECK_EQ(*u21, *u1);
  FAST_CHECK_NE(*u21, *u2);
  FAST_CHECK_NE(*u21, *u3);
  FAST_CHECK_EQ(*u21, *u21);
}

TEST_CASE("Derivative equality")
{
  VariablePtr u = Space(10).createVariable("u");
  VariablePtr u1 = u->subvariable(Space(5), "u1", Space(2));
  VariablePtr du1a = dot(u1);
  VariablePtr du1b = dot(u)->subvariable(Space(5), "du1b", Space(2));

  FAST_CHECK_EQ(*du1a, *du1b);

  VariablePtr v = Space(6, 7, 6).createVariable("v");
  VariablePtr v2 = v->subvariable(Space(3), "v2", Space(3, 4, 3));
  VariablePtr d2v2a = dot(v2, 2);
  VariablePtr d2v2b = dot(v, 2)->subvariable(Space(3), "d2v2b", Space(3, 4, 3));

  FAST_CHECK_EQ(*d2v2a, *d2v2b);
}

TEST_CASE("Subvariable contains")
{
  VariablePtr u = Space(3).createVariable("u");
  VariablePtr v = Space(15).createVariable("v");                // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
  VariablePtr v1 = v->subvariable(Space(7), "v1", Space(1));    //   1 2 3 4 5 6 7
  VariablePtr v2 = v->subvariable(Space(8), "v2", Space(4));    //         4 5 6 7 8 9 10 11
  VariablePtr v3 = v->subvariable(Space(7), "v3", Space(8));    //                 8 9 10 11 12 13 14
  VariablePtr v11 = v1->subvariable(Space(4), "v11");           //   1 2 3 4
  VariablePtr v12 = v1->subvariable(Space(3), "v12", Space(4)); //           5 6 7
  VariablePtr v21 = v2->subvariable(Space(2), "v21", Space(2)); //             6 7
  VariablePtr v22 = v2->subvariable(Space(2), "v22", Space(5)); //                   9 10
  VariablePtr v30 = v3->subvariable(Space(7), "v30");           //                 8 9 10 11 12 13 14

  {
    FAST_CHECK_UNARY(u->isSuperVariable());
    FAST_CHECK_UNARY(v->isSuperVariable());
    FAST_CHECK_UNARY(!v1->isSuperVariable());
    FAST_CHECK_UNARY(!v2->isSuperVariable());
    FAST_CHECK_UNARY(!v3->isSuperVariable());
    FAST_CHECK_UNARY(!v11->isSuperVariable());
    FAST_CHECK_UNARY(!v12->isSuperVariable());
    FAST_CHECK_UNARY(!v11->isSuperVariable());
    FAST_CHECK_UNARY(!v22->isSuperVariable());
    FAST_CHECK_UNARY(!v30->isSuperVariable());
  }
  {
    FAST_CHECK_UNARY(!u->isSuperVariableOf(*u));
    FAST_CHECK_UNARY(!u->isSuperVariableOf(*v));
    FAST_CHECK_UNARY(!u->isSuperVariableOf(*v1));
    FAST_CHECK_UNARY(!u->isSuperVariableOf(*v2));
    FAST_CHECK_UNARY(!u->isSuperVariableOf(*v3));
    FAST_CHECK_UNARY(!u->isSuperVariableOf(*v11));
    FAST_CHECK_UNARY(!u->isSuperVariableOf(*v12));
    FAST_CHECK_UNARY(!u->isSuperVariableOf(*v21));
    FAST_CHECK_UNARY(!u->isSuperVariableOf(*v22));
    FAST_CHECK_UNARY(!u->isSuperVariableOf(*v30));

    FAST_CHECK_UNARY(!v->isSuperVariableOf(*u));
    FAST_CHECK_UNARY(!v->isSuperVariableOf(*v));
    FAST_CHECK_UNARY(v->isSuperVariableOf(*v1));
    FAST_CHECK_UNARY(v->isSuperVariableOf(*v2));
    FAST_CHECK_UNARY(v->isSuperVariableOf(*v3));
    FAST_CHECK_UNARY(v->isSuperVariableOf(*v11));
    FAST_CHECK_UNARY(v->isSuperVariableOf(*v12));
    FAST_CHECK_UNARY(v->isSuperVariableOf(*v21));
    FAST_CHECK_UNARY(v->isSuperVariableOf(*v22));
    FAST_CHECK_UNARY(v->isSuperVariableOf(*v30));

    FAST_CHECK_UNARY(!v1->isSuperVariableOf(*u));
    FAST_CHECK_UNARY(!v1->isSuperVariableOf(*v));
    FAST_CHECK_UNARY(!v1->isSuperVariableOf(*v1));
    FAST_CHECK_UNARY(!v1->isSuperVariableOf(*v2));
    FAST_CHECK_UNARY(!v1->isSuperVariableOf(*v3));
    FAST_CHECK_UNARY(!v1->isSuperVariableOf(*v11));
    FAST_CHECK_UNARY(!v1->isSuperVariableOf(*v12));
    FAST_CHECK_UNARY(!v1->isSuperVariableOf(*v21));
    FAST_CHECK_UNARY(!v1->isSuperVariableOf(*v22));
    FAST_CHECK_UNARY(!v1->isSuperVariableOf(*v30));

    FAST_CHECK_UNARY(!v2->isSuperVariableOf(*u));
    FAST_CHECK_UNARY(!v2->isSuperVariableOf(*v));
    FAST_CHECK_UNARY(!v2->isSuperVariableOf(*v1));
    FAST_CHECK_UNARY(!v2->isSuperVariableOf(*v2));
    FAST_CHECK_UNARY(!v2->isSuperVariableOf(*v3));
    FAST_CHECK_UNARY(!v2->isSuperVariableOf(*v11));
    FAST_CHECK_UNARY(!v2->isSuperVariableOf(*v12));
    FAST_CHECK_UNARY(!v2->isSuperVariableOf(*v21));
    FAST_CHECK_UNARY(!v2->isSuperVariableOf(*v22));
    FAST_CHECK_UNARY(!v2->isSuperVariableOf(*v30));

    FAST_CHECK_UNARY(!v3->isSuperVariableOf(*u));
    FAST_CHECK_UNARY(!v3->isSuperVariableOf(*v));
    FAST_CHECK_UNARY(!v3->isSuperVariableOf(*v1));
    FAST_CHECK_UNARY(!v3->isSuperVariableOf(*v2));
    FAST_CHECK_UNARY(!v3->isSuperVariableOf(*v3));
    FAST_CHECK_UNARY(!v3->isSuperVariableOf(*v11));
    FAST_CHECK_UNARY(!v3->isSuperVariableOf(*v12));
    FAST_CHECK_UNARY(!v3->isSuperVariableOf(*v21));
    FAST_CHECK_UNARY(!v3->isSuperVariableOf(*v22));
    FAST_CHECK_UNARY(!v3->isSuperVariableOf(*v30));

    FAST_CHECK_UNARY(!v11->isSuperVariableOf(*u));
    FAST_CHECK_UNARY(!v11->isSuperVariableOf(*v));
    FAST_CHECK_UNARY(!v11->isSuperVariableOf(*v1));
    FAST_CHECK_UNARY(!v11->isSuperVariableOf(*v2));
    FAST_CHECK_UNARY(!v11->isSuperVariableOf(*v3));
    FAST_CHECK_UNARY(!v11->isSuperVariableOf(*v11));
    FAST_CHECK_UNARY(!v11->isSuperVariableOf(*v12));
    FAST_CHECK_UNARY(!v11->isSuperVariableOf(*v21));
    FAST_CHECK_UNARY(!v11->isSuperVariableOf(*v22));
    FAST_CHECK_UNARY(!v11->isSuperVariableOf(*v30));

    FAST_CHECK_UNARY(!v12->isSuperVariableOf(*u));
    FAST_CHECK_UNARY(!v12->isSuperVariableOf(*v));
    FAST_CHECK_UNARY(!v12->isSuperVariableOf(*v1));
    FAST_CHECK_UNARY(!v12->isSuperVariableOf(*v2));
    FAST_CHECK_UNARY(!v12->isSuperVariableOf(*v3));
    FAST_CHECK_UNARY(!v12->isSuperVariableOf(*v11));
    FAST_CHECK_UNARY(!v12->isSuperVariableOf(*v12));
    FAST_CHECK_UNARY(!v12->isSuperVariableOf(*v21));
    FAST_CHECK_UNARY(!v12->isSuperVariableOf(*v22));
    FAST_CHECK_UNARY(!v12->isSuperVariableOf(*v30));

    FAST_CHECK_UNARY(!v21->isSuperVariableOf(*u));
    FAST_CHECK_UNARY(!v21->isSuperVariableOf(*v));
    FAST_CHECK_UNARY(!v21->isSuperVariableOf(*v1));
    FAST_CHECK_UNARY(!v21->isSuperVariableOf(*v2));
    FAST_CHECK_UNARY(!v21->isSuperVariableOf(*v3));
    FAST_CHECK_UNARY(!v21->isSuperVariableOf(*v11));
    FAST_CHECK_UNARY(!v21->isSuperVariableOf(*v12));
    FAST_CHECK_UNARY(!v21->isSuperVariableOf(*v21));
    FAST_CHECK_UNARY(!v21->isSuperVariableOf(*v22));
    FAST_CHECK_UNARY(!v21->isSuperVariableOf(*v30));

    FAST_CHECK_UNARY(!v22->isSuperVariableOf(*u));
    FAST_CHECK_UNARY(!v22->isSuperVariableOf(*v));
    FAST_CHECK_UNARY(!v22->isSuperVariableOf(*v1));
    FAST_CHECK_UNARY(!v22->isSuperVariableOf(*v2));
    FAST_CHECK_UNARY(!v22->isSuperVariableOf(*v3));
    FAST_CHECK_UNARY(!v22->isSuperVariableOf(*v11));
    FAST_CHECK_UNARY(!v22->isSuperVariableOf(*v12));
    FAST_CHECK_UNARY(!v22->isSuperVariableOf(*v21));
    FAST_CHECK_UNARY(!v22->isSuperVariableOf(*v22));
    FAST_CHECK_UNARY(!v22->isSuperVariableOf(*v30));

    FAST_CHECK_UNARY(!v30->isSuperVariableOf(*u));
    FAST_CHECK_UNARY(!v30->isSuperVariableOf(*v));
    FAST_CHECK_UNARY(!v30->isSuperVariableOf(*v1));
    FAST_CHECK_UNARY(!v30->isSuperVariableOf(*v2));
    FAST_CHECK_UNARY(!v30->isSuperVariableOf(*v3));
    FAST_CHECK_UNARY(!v30->isSuperVariableOf(*v11));
    FAST_CHECK_UNARY(!v30->isSuperVariableOf(*v12));
    FAST_CHECK_UNARY(!v30->isSuperVariableOf(*v21));
    FAST_CHECK_UNARY(!v30->isSuperVariableOf(*v22));
    FAST_CHECK_UNARY(!v30->isSuperVariableOf(*v30));
  }
  {
    FAST_CHECK_UNARY(u->contains(*u));
    FAST_CHECK_UNARY(!u->contains(*v));
    FAST_CHECK_UNARY(!u->contains(*v1));
    FAST_CHECK_UNARY(!u->contains(*v2));
    FAST_CHECK_UNARY(!u->contains(*v3));
    FAST_CHECK_UNARY(!u->contains(*v11));
    FAST_CHECK_UNARY(!u->contains(*v12));
    FAST_CHECK_UNARY(!u->contains(*v21));
    FAST_CHECK_UNARY(!u->contains(*v22));
    FAST_CHECK_UNARY(!u->contains(*v30));

    FAST_CHECK_UNARY(!v->contains(*u));
    FAST_CHECK_UNARY(v->contains(*v));
    FAST_CHECK_UNARY(v->contains(*v1));
    FAST_CHECK_UNARY(v->contains(*v2));
    FAST_CHECK_UNARY(v->contains(*v3));
    FAST_CHECK_UNARY(v->contains(*v11));
    FAST_CHECK_UNARY(v->contains(*v12));
    FAST_CHECK_UNARY(v->contains(*v21));
    FAST_CHECK_UNARY(v->contains(*v22));
    FAST_CHECK_UNARY(v->contains(*v30));

    FAST_CHECK_UNARY(!v1->contains(*u));
    FAST_CHECK_UNARY(!v1->contains(*v));
    FAST_CHECK_UNARY(v1->contains(*v1));
    FAST_CHECK_UNARY(!v1->contains(*v2));
    FAST_CHECK_UNARY(!v1->contains(*v3));
    FAST_CHECK_UNARY(v1->contains(*v11));
    FAST_CHECK_UNARY(v1->contains(*v12));
    FAST_CHECK_UNARY(v1->contains(*v21));
    FAST_CHECK_UNARY(!v1->contains(*v22));
    FAST_CHECK_UNARY(!v1->contains(*v30));

    FAST_CHECK_UNARY(!v2->contains(*u));
    FAST_CHECK_UNARY(!v2->contains(*v));
    FAST_CHECK_UNARY(!v2->contains(*v1));
    FAST_CHECK_UNARY(v2->contains(*v2));
    FAST_CHECK_UNARY(!v2->contains(*v3));
    FAST_CHECK_UNARY(!v2->contains(*v11));
    FAST_CHECK_UNARY(v2->contains(*v12));
    FAST_CHECK_UNARY(v2->contains(*v21));
    FAST_CHECK_UNARY(v2->contains(*v22));
    FAST_CHECK_UNARY(!v2->contains(*v30));

    FAST_CHECK_UNARY(!v3->contains(*u));
    FAST_CHECK_UNARY(!v3->contains(*v));
    FAST_CHECK_UNARY(!v3->contains(*v1));
    FAST_CHECK_UNARY(!v3->contains(*v2));
    FAST_CHECK_UNARY(v3->contains(*v3));
    FAST_CHECK_UNARY(!v3->contains(*v11));
    FAST_CHECK_UNARY(!v3->contains(*v12));
    FAST_CHECK_UNARY(!v3->contains(*v21));
    FAST_CHECK_UNARY(v3->contains(*v22));
    FAST_CHECK_UNARY(v3->contains(*v30));

    FAST_CHECK_UNARY(!v11->contains(*u));
    FAST_CHECK_UNARY(!v11->contains(*v));
    FAST_CHECK_UNARY(!v11->contains(*v1));
    FAST_CHECK_UNARY(!v11->contains(*v2));
    FAST_CHECK_UNARY(!v11->contains(*v3));
    FAST_CHECK_UNARY(v11->contains(*v11));
    FAST_CHECK_UNARY(!v11->contains(*v12));
    FAST_CHECK_UNARY(!v11->contains(*v21));
    FAST_CHECK_UNARY(!v11->contains(*v22));
    FAST_CHECK_UNARY(!v11->contains(*v30));

    FAST_CHECK_UNARY(!v12->contains(*u));
    FAST_CHECK_UNARY(!v12->contains(*v));
    FAST_CHECK_UNARY(!v12->contains(*v1));
    FAST_CHECK_UNARY(!v12->contains(*v2));
    FAST_CHECK_UNARY(!v12->contains(*v3));
    FAST_CHECK_UNARY(!v12->contains(*v11));
    FAST_CHECK_UNARY(v12->contains(*v12));
    FAST_CHECK_UNARY(v12->contains(*v21));
    FAST_CHECK_UNARY(!v12->contains(*v22));
    FAST_CHECK_UNARY(!v12->contains(*v30));

    FAST_CHECK_UNARY(!v21->contains(*u));
    FAST_CHECK_UNARY(!v21->contains(*v));
    FAST_CHECK_UNARY(!v21->contains(*v1));
    FAST_CHECK_UNARY(!v21->contains(*v2));
    FAST_CHECK_UNARY(!v21->contains(*v3));
    FAST_CHECK_UNARY(!v21->contains(*v11));
    FAST_CHECK_UNARY(!v21->contains(*v12));
    FAST_CHECK_UNARY(v21->contains(*v21));
    FAST_CHECK_UNARY(!v21->contains(*v22));
    FAST_CHECK_UNARY(!v21->contains(*v30));

    FAST_CHECK_UNARY(!v22->contains(*u));
    FAST_CHECK_UNARY(!v22->contains(*v));
    FAST_CHECK_UNARY(!v22->contains(*v1));
    FAST_CHECK_UNARY(!v22->contains(*v2));
    FAST_CHECK_UNARY(!v22->contains(*v3));
    FAST_CHECK_UNARY(!v22->contains(*v11));
    FAST_CHECK_UNARY(!v22->contains(*v12));
    FAST_CHECK_UNARY(!v22->contains(*v21));
    FAST_CHECK_UNARY(v22->contains(*v22));
    FAST_CHECK_UNARY(!v22->contains(*v30));

    FAST_CHECK_UNARY(!v30->contains(*u));
    FAST_CHECK_UNARY(!v30->contains(*v));
    FAST_CHECK_UNARY(!v30->contains(*v1));
    FAST_CHECK_UNARY(!v30->contains(*v2));
    FAST_CHECK_UNARY(v30->contains(*v3));
    FAST_CHECK_UNARY(!v30->contains(*v11));
    FAST_CHECK_UNARY(!v30->contains(*v12));
    FAST_CHECK_UNARY(!v30->contains(*v21));
    FAST_CHECK_UNARY(v30->contains(*v22));
    FAST_CHECK_UNARY(v30->contains(*v30));
  }

  {
    FAST_CHECK_UNARY(u->intersects(*u));
    FAST_CHECK_UNARY(!u->intersects(*v));
    FAST_CHECK_UNARY(!u->intersects(*v1));
    FAST_CHECK_UNARY(!u->intersects(*v2));
    FAST_CHECK_UNARY(!u->intersects(*v3));
    FAST_CHECK_UNARY(!u->intersects(*v11));
    FAST_CHECK_UNARY(!u->intersects(*v12));
    FAST_CHECK_UNARY(!u->intersects(*v21));
    FAST_CHECK_UNARY(!u->intersects(*v22));
    FAST_CHECK_UNARY(!u->intersects(*v30));

    FAST_CHECK_UNARY(!v->intersects(*u));
    FAST_CHECK_UNARY(v->intersects(*v));
    FAST_CHECK_UNARY(v->intersects(*v1));
    FAST_CHECK_UNARY(v->intersects(*v2));
    FAST_CHECK_UNARY(v->intersects(*v3));
    FAST_CHECK_UNARY(v->intersects(*v11));
    FAST_CHECK_UNARY(v->intersects(*v12));
    FAST_CHECK_UNARY(v->intersects(*v21));
    FAST_CHECK_UNARY(v->intersects(*v22));
    FAST_CHECK_UNARY(v->intersects(*v30));

    FAST_CHECK_UNARY(!v1->intersects(*u));
    FAST_CHECK_UNARY(v1->intersects(*v));
    FAST_CHECK_UNARY(v1->intersects(*v1));
    FAST_CHECK_UNARY(v1->intersects(*v2));
    FAST_CHECK_UNARY(!v1->intersects(*v3));
    FAST_CHECK_UNARY(v1->intersects(*v11));
    FAST_CHECK_UNARY(v1->intersects(*v12));
    FAST_CHECK_UNARY(v1->intersects(*v21));
    FAST_CHECK_UNARY(!v1->intersects(*v22));
    FAST_CHECK_UNARY(!v1->intersects(*v30));

    FAST_CHECK_UNARY(!v2->intersects(*u));
    FAST_CHECK_UNARY(v2->intersects(*v));
    FAST_CHECK_UNARY(v2->intersects(*v1));
    FAST_CHECK_UNARY(v2->intersects(*v2));
    FAST_CHECK_UNARY(v2->intersects(*v3));
    FAST_CHECK_UNARY(v2->intersects(*v11));
    FAST_CHECK_UNARY(v2->intersects(*v12));
    FAST_CHECK_UNARY(v2->intersects(*v21));
    FAST_CHECK_UNARY(v2->intersects(*v22));
    FAST_CHECK_UNARY(v2->intersects(*v30));

    FAST_CHECK_UNARY(!v3->intersects(*u));
    FAST_CHECK_UNARY(v3->intersects(*v));
    FAST_CHECK_UNARY(!v3->intersects(*v1));
    FAST_CHECK_UNARY(v3->intersects(*v2));
    FAST_CHECK_UNARY(v3->intersects(*v3));
    FAST_CHECK_UNARY(!v3->intersects(*v11));
    FAST_CHECK_UNARY(!v3->intersects(*v12));
    FAST_CHECK_UNARY(!v3->intersects(*v21));
    FAST_CHECK_UNARY(v3->intersects(*v22));
    FAST_CHECK_UNARY(v3->intersects(*v30));

    FAST_CHECK_UNARY(!v11->intersects(*u));
    FAST_CHECK_UNARY(v11->intersects(*v));
    FAST_CHECK_UNARY(v11->intersects(*v1));
    FAST_CHECK_UNARY(v11->intersects(*v2));
    FAST_CHECK_UNARY(!v11->intersects(*v3));
    FAST_CHECK_UNARY(v11->intersects(*v11));
    FAST_CHECK_UNARY(!v11->intersects(*v12));
    FAST_CHECK_UNARY(!v11->intersects(*v21));
    FAST_CHECK_UNARY(!v11->intersects(*v22));
    FAST_CHECK_UNARY(!v11->intersects(*v30));

    FAST_CHECK_UNARY(!v12->intersects(*u));
    FAST_CHECK_UNARY(v12->intersects(*v));
    FAST_CHECK_UNARY(v12->intersects(*v1));
    FAST_CHECK_UNARY(v12->intersects(*v2));
    FAST_CHECK_UNARY(!v12->intersects(*v3));
    FAST_CHECK_UNARY(!v12->intersects(*v11));
    FAST_CHECK_UNARY(v12->intersects(*v12));
    FAST_CHECK_UNARY(v12->intersects(*v21));
    FAST_CHECK_UNARY(!v12->intersects(*v22));
    FAST_CHECK_UNARY(!v12->intersects(*v30));

    FAST_CHECK_UNARY(!v21->intersects(*u));
    FAST_CHECK_UNARY(v21->intersects(*v));
    FAST_CHECK_UNARY(v21->intersects(*v1));
    FAST_CHECK_UNARY(v21->intersects(*v2));
    FAST_CHECK_UNARY(!v21->intersects(*v3));
    FAST_CHECK_UNARY(!v21->intersects(*v11));
    FAST_CHECK_UNARY(v21->intersects(*v12));
    FAST_CHECK_UNARY(v21->intersects(*v21));
    FAST_CHECK_UNARY(!v21->intersects(*v22));
    FAST_CHECK_UNARY(!v21->intersects(*v30));

    FAST_CHECK_UNARY(!v22->intersects(*u));
    FAST_CHECK_UNARY(v22->intersects(*v));
    FAST_CHECK_UNARY(!v22->intersects(*v1));
    FAST_CHECK_UNARY(v22->intersects(*v2));
    FAST_CHECK_UNARY(v22->intersects(*v3));
    FAST_CHECK_UNARY(!v22->intersects(*v11));
    FAST_CHECK_UNARY(!v22->intersects(*v12));
    FAST_CHECK_UNARY(!v22->intersects(*v21));
    FAST_CHECK_UNARY(v22->intersects(*v22));
    FAST_CHECK_UNARY(v22->intersects(*v30));

    FAST_CHECK_UNARY(!v30->intersects(*u));
    FAST_CHECK_UNARY(v30->intersects(*v));
    FAST_CHECK_UNARY(!v30->intersects(*v1));
    FAST_CHECK_UNARY(v30->intersects(*v2));
    FAST_CHECK_UNARY(v30->intersects(*v3));
    FAST_CHECK_UNARY(!v30->intersects(*v11));
    FAST_CHECK_UNARY(!v30->intersects(*v12));
    FAST_CHECK_UNARY(!v30->intersects(*v21));
    FAST_CHECK_UNARY(v30->intersects(*v22));
    FAST_CHECK_UNARY(v30->intersects(*v30));
  }
}
