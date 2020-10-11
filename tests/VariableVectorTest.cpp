/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Range.h>
#include <tvm/Variable.h>
#include <tvm/VariableVector.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm;

void testCreation(VariablePtr v1, VariablePtr v2, VariablePtr v3, VariablePtr v4)
{
  VariableVector vv1;
  vv1.add(v2);
  vv1.add(v3);
  vv1.add(v1);

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
  vv1.clear();
  FAST_CHECK_EQ(vv1.numberOfVariables(), 0);
  FAST_CHECK_EQ(vv1.totalSize(), 0);

  VariableVector vv2({v1, v2, v3});
  FAST_CHECK_EQ(vv2.numberOfVariables(), 3);
  FAST_CHECK_EQ(vv2.totalSize(), 9);
  FAST_CHECK_EQ(vv2[0], v1);
  FAST_CHECK_EQ(vv2[1], v2);
  FAST_CHECK_EQ(vv2[2], v3);

  CHECK(vv2.add(v1) == false);
  CHECK(vv2.remove(*v4) == false);

  std::vector<VariablePtr> vec = {v1, v2};
  std::vector<VariablePtr> vec2 = {v1, v3, v4};
  VariableVector vv3(vec);
  for(const auto & v : vec)
  {
    vv3.add(v);
  }
  FAST_CHECK_EQ(vv3.numberOfVariables(), 2);
  FAST_CHECK_EQ(vv3.totalSize(), 7);
  FAST_CHECK_EQ(vv3[0], v1);
  FAST_CHECK_EQ(vv3[1], v2);
  for(const auto & v : vec2)
  {
    vv3.add(v);
  }
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

  VariableVector vv4(v3, vv3, vec2);
  FAST_CHECK_EQ(vv4.numberOfVariables(), 4);
  FAST_CHECK_EQ(vv4.totalSize(), 12);
  FAST_CHECK_EQ(vv4[0], v3);
  FAST_CHECK_EQ(vv4[1], v1);
  FAST_CHECK_EQ(vv4[2], v2);
  FAST_CHECK_EQ(vv4[3], v4);
}

TEST_CASE("Test VariableVector creation 1")
{
  VariablePtr v1 = Space(3).createVariable("v1");
  VariablePtr v2 = Space(4).createVariable("v2");
  VariablePtr v3 = Space(2).createVariable("v3");
  VariablePtr v4 = Space(3).createVariable("v4");

  testCreation(v1, v2, v3, v4);
}

TEST_CASE("Test VariableVector creation 2")
{
  VariablePtr v1_ = Space(7).createVariable("v1_");
  VariablePtr v2 = Space(4).createVariable("v2");
  VariablePtr v3_ = Space(5).createVariable("v3_");
  VariablePtr v4_ = Space(5).createVariable("v4_");

  VariablePtr v1 = v1_->subvariable(Space(3), "v1", Space(2));
  VariablePtr v3__ = v3_->subvariable(Space(3), "v3__", Space(0));
  VariablePtr v3 = v3_->subvariable(Space(2), "v3", Space(1));
  VariablePtr v4 = v4_->subvariable(Space(3), "v4", Space(0));

  testCreation(v1, v2, v3, v4);
}

TEST_CASE("Test VariableVector creation 3")
{
  VariablePtr u = Space(8).createVariable("u");
  VariablePtr v = Space(4).createVariable("v");
  VariablePtr w = Space(7).createVariable("w");
  VariablePtr u0 = u->subvariable(Space(8), "u0");            // u0 == u
  VariablePtr u1 = u->subvariable(Space(4), "u1", Space(0));
  VariablePtr u2 = u->subvariable(Space(4), "u2", Space(2));
  VariablePtr u3 = u->subvariable(Space(2), "u3", Space(5));
  VariablePtr u4 = u0->subvariable(Space(4), "u4", Space(2)); // u4 == u2

  VariableVector vv;
  FAST_CHECK_UNARY(vv.add(u));
  FAST_CHECK_UNARY(vv.add(v));
  FAST_CHECK_UNARY(vv.add(w));
  FAST_CHECK_UNARY_FALSE(vv.add(u));
  FAST_CHECK_UNARY_FALSE(vv.add(u0));
  FAST_CHECK_UNARY_FALSE(vv.add(u1));
  FAST_CHECK_UNARY_FALSE(vv.add(u2));
  FAST_CHECK_UNARY_FALSE(vv.add(u3));
  FAST_CHECK_UNARY_FALSE(vv.add(u4));
  FAST_CHECK_UNARY_FALSE(vv.remove(*u1));
  FAST_CHECK_UNARY_FALSE(vv.remove(*u2));
  FAST_CHECK_UNARY_FALSE(vv.remove(*u3));
  FAST_CHECK_UNARY_FALSE(vv.remove(*u4));
  FAST_CHECK_UNARY(vv.remove(*u0));         // This removes u
  FAST_CHECK_EQ(vv.numberOfVariables(), 2);
  FAST_CHECK_EQ(vv.totalSize(), 11);

  FAST_CHECK_UNARY(vv.add(u2));
  CHECK_THROWS(vv.add(u1));
  FAST_CHECK_UNARY_FALSE(vv.add(u4));
  FAST_CHECK_UNARY(vv.remove(*u4));         //This removes u2
  FAST_CHECK_UNARY(vv.remove(*w));
  CHECK_NOTHROW(vv.add(u1));
  FAST_CHECK_UNARY(vv.add(w));
  CHECK_NOTHROW(vv.add(u3));
}

TEST_CASE("addAndGetIndex")
{
  VariablePtr u = Space(8).createVariable("u");
  VariablePtr v = Space(4).createVariable("v");
  VariablePtr w = Space(7).createVariable("w");
  VariablePtr u0 = u->subvariable(Space(8), "u0"); // u0 == u
  VariablePtr u1 = u->subvariable(Space(4), "u1", Space(0));
  VariablePtr u2 = u->subvariable(Space(4), "u2", Space(2));
  VariablePtr u3 = u->subvariable(Space(2), "u3", Space(5));
  VariablePtr u4 = u0->subvariable(Space(4), "u4", Space(2)); // u4 == u2

  VariableVector vv;
  FAST_CHECK_EQ(vv.addAndGetIndex(u), 0);
  FAST_CHECK_EQ(vv.addAndGetIndex(v), 1);
  FAST_CHECK_EQ(vv.addAndGetIndex(u0), 0);
  FAST_CHECK_EQ(vv.addAndGetIndex(u0, true), 0);
  FAST_CHECK_EQ(vv.addAndGetIndex(u1), -1);
  FAST_CHECK_EQ(vv.addAndGetIndex(u2, true), 0);
  vv.remove(0);   // remove u
  FAST_CHECK_EQ(vv.addAndGetIndex(u2), 1);
  FAST_CHECK_EQ(vv.addAndGetIndex(u4), 1);
  FAST_CHECK_EQ(vv.addAndGetIndex(u4, true), 1);
  CHECK_THROWS(vv.addAndGetIndex(u3));
  CHECK_THROWS(vv.addAndGetIndex(u3, true));
  FAST_CHECK_UNARY(vv.remove(*u4)); //remove u2 (==u4)
  FAST_CHECK_EQ(vv.addAndGetIndex(u1), 1);
  FAST_CHECK_EQ(vv.addAndGetIndex(w), 2);
  FAST_CHECK_EQ(vv.addAndGetIndex(u3), 3);
  FAST_CHECK_EQ(vv.numberOfVariables(), 4);   //v u1 w u3
  FAST_CHECK_EQ(vv.totalSize(), 17);
}

void testMapping(VariablePtr v1, VariablePtr v2, VariablePtr v3, VariablePtr v4)
{
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

  FAST_CHECK_EQ(v1->getMappingIn(vv1), Range{0, 3});
  FAST_CHECK_EQ(v2->getMappingIn(vv1), Range{3, 4});
  FAST_CHECK_EQ(v3->getMappingIn(vv1), Range{7, 2});
  CHECK_THROWS(v4->getMappingIn(vv1));
  FAST_CHECK_EQ(v1->getMappingIn(vv2), Range{6, 3});
  FAST_CHECK_EQ(v2->getMappingIn(vv2), Range{2, 4});
  FAST_CHECK_EQ(v3->getMappingIn(vv2), Range{0, 2});
  CHECK_THROWS(v4->getMappingIn(vv2));

  FAST_CHECK_EQ(vv1.stamp(), s + 3);
  FAST_CHECK_EQ(vv2.stamp(), s + 7);

  vv1.add(v4);
  FAST_CHECK_EQ(vv1.stamp(), s + 8);
  FAST_CHECK_EQ(v1->getMappingIn(vv1), Range{0, 3});
  FAST_CHECK_EQ(v2->getMappingIn(vv1), Range{3, 4});
  FAST_CHECK_EQ(v3->getMappingIn(vv1), Range{7, 2});
  FAST_CHECK_EQ(v4->getMappingIn(vv1), Range{9, 3});
  vv1.remove(*v2);
  FAST_CHECK_EQ(vv1.stamp(), s + 9);
  FAST_CHECK_EQ(v1->getMappingIn(vv1), Range{0, 3});
  CHECK_THROWS(v2->getMappingIn(vv1));
  FAST_CHECK_EQ(v3->getMappingIn(vv1), Range{3, 2});
  FAST_CHECK_EQ(v4->getMappingIn(vv1), Range{5, 3});
  FAST_CHECK_EQ(v1->getMappingIn(vv2), Range{6, 3});
  FAST_CHECK_EQ(v2->getMappingIn(vv2), Range{2, 4});
  FAST_CHECK_EQ(v3->getMappingIn(vv2), Range{0, 2});
  CHECK_THROWS(v4->getMappingIn(vv2));
}

TEST_CASE("Test Mapping 1")
{
  VariablePtr v1 = Space(3).createVariable("v1");
  VariablePtr v2 = Space(4).createVariable("v2");
  VariablePtr v3 = Space(2).createVariable("v3");
  VariablePtr v4 = Space(3).createVariable("v4");

  testMapping(v1, v2, v3, v4);
}

TEST_CASE("Test Mapping 2")
{
  VariablePtr v1_ = Space(7).createVariable("v1_");
  VariablePtr v2 = Space(4).createVariable("v2");
  VariablePtr v3_ = Space(5).createVariable("v3_");
  VariablePtr v4_ = Space(5).createVariable("v4_");

  VariablePtr v1 = v1_->subvariable(Space(3), "v1", Space(2));
  VariablePtr v3__ = v3_->subvariable(Space(3), "v3__", Space(0));
  VariablePtr v3 = v3_->subvariable(Space(2), "v3", Space(1));
  VariablePtr v4 = v4_->subvariable(Space(3), "v4", Space(0));

  testMapping(v1, v2, v3, v4);
}

TEST_CASE("Test Mapping 3") 
{
  VariablePtr u = Space(8).createVariable("u");
  VariablePtr v = Space(10).createVariable("v");
  VariablePtr w = Space(7).createVariable("w");

  VariablePtr u1 = u->subvariable(Space(3), "u1", Space(0));
  VariablePtr u2 = u->subvariable(Space(3), "u2", Space(3));
  VariablePtr u3 = u->subvariable(Space(2), "u3", Space(6));
  VariablePtr v1 = v->subvariable(Space(7), "v1", Space(1));
  VariablePtr v2 = v->subvariable(Space(5), "v2", Space(5));
  VariablePtr v11 = v1->subvariable(Space(3), "v11", Space(0));
  VariablePtr v12 = v1->subvariable(Space(3), "v12", Space(4));

  VariableVector vv1(u,v);
  FAST_CHECK_EQ(u1->getMappingIn(vv1), Range(0,3));
  FAST_CHECK_EQ(u2->getMappingIn(vv1), Range(3,3));
  FAST_CHECK_EQ(u3->getMappingIn(vv1), Range(6,2));
  FAST_CHECK_EQ(v1->getMappingIn(vv1), Range(9,7));
  FAST_CHECK_EQ(v2->getMappingIn(vv1), Range(13,5));
  FAST_CHECK_EQ(v11->getMappingIn(vv1), Range(9,3));
  FAST_CHECK_EQ(v12->getMappingIn(vv1), Range(13,3));

  //size:            2   7  5   3
  VariableVector vv2(u3, w, v2, u1);
  CHECK_THROWS(u->getMappingIn(vv2));
  CHECK_THROWS(v->getMappingIn(vv2));
  FAST_CHECK_EQ(w->getMappingIn(vv2), Range(2,7));
  FAST_CHECK_EQ(u1->getMappingIn(vv2), Range(14,3));
  CHECK_THROWS(u2->getMappingIn(vv2));
  FAST_CHECK_EQ(u3->getMappingIn(vv2), Range(0,2));
  CHECK_THROWS(v1->getMappingIn(vv2));
  FAST_CHECK_EQ(v2->getMappingIn(vv2), Range(9,5));
  CHECK_THROWS(v11->getMappingIn(vv2));
  FAST_CHECK_EQ(v12->getMappingIn(vv2), Range(9,3));

  vv2.add(u2);
  FAST_CHECK_EQ(u2->getMappingIn(vv2), Range(17,3));

  vv2.remove(1);
  CHECK_THROWS(u->getMappingIn(vv2));
  CHECK_THROWS(v->getMappingIn(vv2));
  CHECK_THROWS(w->getMappingIn(vv2));
  FAST_CHECK_EQ(u1->getMappingIn(vv2), Range(7,3));
  FAST_CHECK_EQ(u2->getMappingIn(vv2), Range(10,3));
  FAST_CHECK_EQ(u3->getMappingIn(vv2), Range(0,2));
  CHECK_THROWS(v1->getMappingIn(vv2));
  FAST_CHECK_EQ(v2->getMappingIn(vv2), Range(2,5));
  CHECK_THROWS(v11->getMappingIn(vv2));
  FAST_CHECK_EQ(v12->getMappingIn(vv2), Range(2,3));
}

void testDerivation(VariablePtr v1, VariablePtr v2, VariablePtr v3)
{
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

TEST_CASE("Test derivation 1")
{
  VariablePtr v1 = Space(3).createVariable("v1");
  VariablePtr v2 = Space(4).createVariable("v2");
  VariablePtr v3 = Space(2).createVariable("v3");

  testDerivation(v1, v2, v3);
}

TEST_CASE("Test derivation 2")
{
  VariablePtr v1_ = Space(7).createVariable("v1_");
  VariablePtr v2 = Space(4).createVariable("v2");
  VariablePtr v3_ = Space(5).createVariable("v3_");

  VariablePtr v1 = v1_->subvariable(Space(3), "v1", Space(2));
  VariablePtr v3__ = v3_->subvariable(Space(3), "v3__", Space(0));
  VariablePtr v3 = v3_->subvariable(Space(2), "v3", Space(1));

  testDerivation(v1, v2, v3);
}