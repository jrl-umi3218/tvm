/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Range.h>
#include <tvm/Variable.h>
#include <tvm/VariableVector.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm;

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

  VariablePtr v = Space(3, 4, 3).createVariable("v");
  FAST_CHECK_EQ(v->size(), 4);
  FAST_CHECK_EQ(dot(v)->size(), 3);
  FAST_CHECK_EQ(dot(v,4)->size(), 3);
  FAST_CHECK_UNARY(!v->space().isEuclidean());
  FAST_CHECK_EQ(v->space().size(), 3);
  FAST_CHECK_EQ(v->space().rSize(), 4);
  FAST_CHECK_EQ(v->space().tSize(), 3);
  FAST_CHECK_UNARY_FALSE(v->isEuclidean());
  FAST_CHECK_UNARY(dot(v)->isEuclidean());

  VariablePtr w = v->duplicate("w");
  FAST_CHECK_NE(v, w);
  FAST_CHECK_EQ(v->space(), w->space());
  FAST_CHECK_UNARY_FALSE(w->isEuclidean());
  FAST_CHECK_UNARY(dot(w)->isEuclidean());
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
    Eigen::Vector3d val(1,2,3);
    v << val;
    FAST_CHECK_UNARY(v->value().isApprox(val));
  }
  {
    VariablePtr v = Space(3).createVariable("v");
    Eigen::VectorXd val(5); val << 1, 2, 3, 4, 5;
    v << val.head(3);
    FAST_CHECK_UNARY(v->value().isApprox(Eigen::Vector3d(1, 2, 3)));
  }
  {
    VariablePtr v = Space(3).createVariable("v");
    v << 1, 2, 3;
    FAST_CHECK_UNARY(v->value().isApprox(Eigen::Vector3d(1,2,3)));
  }
  {
    VariablePtr v = Space(3).createVariable("v");
    Eigen::VectorXd val(5); val << 1, 2, 3, 4, 5;
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

TEST_CASE("Test VariableVector creation")
{
  VariablePtr v1 = Space(3).createVariable("v1");
  VariablePtr v2 = Space(4).createVariable("v2");
  VariablePtr v3 = Space(2).createVariable("v3");
  VariablePtr v4 = Space(3).createVariable("v4");

  VariableVector vv1;
  vv1.add(v2);
  vv1.add(v3);
  vv1.add(v1);

  int i1, i2, i3, i4;
  FAST_CHECK_EQ(vv1.numberOfVariables(), 3);
  FAST_CHECK_EQ(vv1.totalSize(), 9);
  FAST_CHECK_EQ(vv1[0], v2);
  FAST_CHECK_EQ(vv1[1], v3);
  FAST_CHECK_EQ(vv1[2], v1);
  FAST_CHECK_UNARY(vv1.contains(*v1));
  FAST_CHECK_UNARY(vv1.contains(*v2));
  FAST_CHECK_UNARY(vv1.contains(*v3));
  FAST_CHECK_UNARY(!vv1.contains(*v4));
  FAST_CHECK_EQ(vv1.indexOf(*v1), 2);
  FAST_CHECK_EQ(vv1.indexOf(*v2), 0);
  FAST_CHECK_EQ(vv1.indexOf(*v3), 1);
  FAST_CHECK_EQ(vv1.indexOf(*v4), -1);

  vv1.remove(*v3);
  FAST_CHECK_EQ(vv1.numberOfVariables(), 2);
  FAST_CHECK_EQ(vv1.totalSize(), 7);
  FAST_CHECK_EQ(vv1[0], v2);
  FAST_CHECK_EQ(vv1[1], v1);
  FAST_CHECK_UNARY(vv1.contains(*v1));
  FAST_CHECK_UNARY(vv1.contains(*v2));
  FAST_CHECK_UNARY(!vv1.contains(*v3));
  FAST_CHECK_UNARY(!vv1.contains(*v4));
  FAST_CHECK_EQ(vv1.indexOf(*v1), 1);
  FAST_CHECK_EQ(vv1.indexOf(*v2), 0);
  FAST_CHECK_EQ(vv1.indexOf(*v3), -1);
  FAST_CHECK_EQ(vv1.indexOf(*v4), -1);

  VariableVector vv2({ v1, v2, v3 });
  FAST_CHECK_EQ(vv2.numberOfVariables(), 3);
  FAST_CHECK_EQ(vv2.totalSize(), 9);
  FAST_CHECK_EQ(vv2[0], v1);
  FAST_CHECK_EQ(vv2[1], v2);
  FAST_CHECK_EQ(vv2[2], v3);

  CHECK(vv2.add(v1) == false);
  CHECK(vv2.remove(*v4) == false);

  std::vector<VariablePtr> vec = { v1, v2 };
  std::vector<VariablePtr> vec2 = { v1, v3, v4 };
  VariableVector vv3(vec);
  for(const auto & v : vec) { vv3.add(v); }
  FAST_CHECK_EQ(vv3.numberOfVariables(), 2);
  FAST_CHECK_EQ(vv3.totalSize(), 7);
  FAST_CHECK_EQ(vv3[0], v1);
  FAST_CHECK_EQ(vv3[1], v2);
  for(const auto & v : vec2) { vv3.add(v); }
  FAST_CHECK_EQ(vv3.numberOfVariables(), 4);
  FAST_CHECK_EQ(vv3.totalSize(), 12);
  FAST_CHECK_EQ(vv3[0], v1);
  FAST_CHECK_EQ(vv3[1], v2);
  FAST_CHECK_EQ(vv3[2], v3);
  FAST_CHECK_EQ(vv3[3], v4);
  FAST_CHECK_UNARY(vv3.contains(*v1));
  FAST_CHECK_UNARY(vv3.contains(*v2));
  FAST_CHECK_UNARY(vv3.contains(*v3));
  FAST_CHECK_UNARY(vv3.contains(*v4));
}

TEST_CASE("Test Mapping")
{
  VariablePtr v1 = Space(3).createVariable("v1");
  VariablePtr v2 = Space(4).createVariable("v2");
  VariablePtr v3 = Space(2).createVariable("v3");
  VariablePtr v4 = Space(3).createVariable("v4");

  VariableVector vv1;
  int s = vv1.stamp();
  vv1.add(v1);
  vv1.add(v2);
  vv1.add(v3);

  VariableVector vv2;
  vv2.add(v3);
  vv2.add(v2);
  vv2.add(v1);

  FAST_CHECK_EQ(vv1.stamp(), s + 3);
  FAST_CHECK_EQ(vv2.stamp(), s + 7);

  FAST_CHECK_EQ(v1->getMappingIn(vv1), Range{ 0, 3 });
  FAST_CHECK_EQ(v2->getMappingIn(vv1), Range{ 3, 4 });
  FAST_CHECK_EQ(v3->getMappingIn(vv1), Range{ 7, 2 });
  CHECK_THROWS(v4->getMappingIn(vv1));
  FAST_CHECK_EQ(v1->getMappingIn(vv2), Range{ 6, 3 });
  FAST_CHECK_EQ(v2->getMappingIn(vv2), Range{ 2, 4 });
  FAST_CHECK_EQ(v3->getMappingIn(vv2), Range{ 0, 2 });
  CHECK_THROWS(v4->getMappingIn(vv2));

  FAST_CHECK_EQ(vv1.stamp(), s + 3);
  FAST_CHECK_EQ(vv2.stamp(), s + 7);

  vv1.add(v4);
  FAST_CHECK_EQ(vv1.stamp(), s + 8);
  FAST_CHECK_EQ(v1->getMappingIn(vv1), Range{ 0, 3 });
  FAST_CHECK_EQ(v2->getMappingIn(vv1), Range{ 3, 4 });
  FAST_CHECK_EQ(v3->getMappingIn(vv1), Range{ 7, 2 });
  FAST_CHECK_EQ(v4->getMappingIn(vv1), Range{ 9, 3 });
  vv1.remove(*v2);
  FAST_CHECK_EQ(vv1.stamp(), s + 9);
  FAST_CHECK_EQ(v1->getMappingIn(vv1), Range{ 0, 3 });
  CHECK_THROWS(v2->getMappingIn(vv1));
  FAST_CHECK_EQ(v3->getMappingIn(vv1), Range{ 3, 2 });
  FAST_CHECK_EQ(v4->getMappingIn(vv1), Range{ 5, 3 });
  FAST_CHECK_EQ(v1->getMappingIn(vv2), Range{ 6, 3 });
  FAST_CHECK_EQ(v2->getMappingIn(vv2), Range{ 2, 4 });
  FAST_CHECK_EQ(v3->getMappingIn(vv2), Range{ 0, 2 });
  CHECK_THROWS(v4->getMappingIn(vv2));
}

TEST_CASE("Test VariableVector derivation")
{
  VariablePtr v1 = Space(3).createVariable("v1");
  VariablePtr v2 = Space(4).createVariable("v2");
  VariablePtr v3 = Space(2).createVariable("v3");

  VariableVector vv;
  vv.add(v1);
  vv.add(v2);
  vv.add(v3);

  auto dvv = dot(vv);
  FAST_CHECK_EQ(dvv[0], dot(v1));
  FAST_CHECK_EQ(dvv[1], dot(v2));
  FAST_CHECK_EQ(dvv[2], dot(v3));

  auto dvv3 = dot(vv, 3);
  FAST_CHECK_EQ(dvv3[0], dot(v1, 3));
  FAST_CHECK_EQ(dvv3[1], dot(v2, 3));
  FAST_CHECK_EQ(dvv3[2], dot(v3, 3));

  auto dvv3b = dot(dvv, 2);
  FAST_CHECK_EQ(dvv3b[0], dot(v1, 3));
  FAST_CHECK_EQ(dvv3b[1], dot(v2, 3));
  FAST_CHECK_EQ(dvv3b[2], dot(v3, 3));
}
