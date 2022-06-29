/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Variable.h>

#include <tvm/internal/RangeCounting.h>
#include <tvm/internal/VariableCountingVector.h>
#include <tvm/internal/VariableVectorPartition.h>

#include <algorithm>
#include <map>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm;
using namespace tvm::internal;

// Given a list of Limit, return the depth of each individual integer
// e.g. for {(2,+), (3,+), (4,-), (6,-), (8,+), (10,-)} returns
//                 {1, 2, 1, 1, 0, 0, 1, 1}
//                  ^  ^  ^  ^  ^  ^  ^  ^
// corresponding to 2  3  4  5  6  7  8  9
std::vector<int> depth(const std::list<RangeCounting::Limit> & limits)
{
  int depth = 0;
  std::vector<int> ret;
  for(auto it = limits.cbegin(); it != std::prev(limits.cend()); ++it)
  {
    depth -= it->type_;

    for(int i = it->i_; i < std::next(it)->i_; ++i)
      ret.push_back(depth);
  }

  return ret;
}

void addRange(std::unordered_map<int, int> & count, const Range & r)
{
  for(int i = r.start; i < r.start + r.dim; ++i)
  {
    if(auto it = count.find(i); it != count.end())
      ++it->second;
    else
      count[i] = 1;
  }
}

void removeRange(std::unordered_map<int, int> & count, const Range & r)
{
  for(int i = r.start; i < r.start + r.dim; ++i)
  {
    if(auto it = count.find(i); it != count.end())
      --it->second;
    else
      count[i] = -1;
  }
}

// Check if count[i] does not exist or count[i]==0;
bool isZeroOrAbsent(const std::unordered_map<int, int> & count, int i)
{
  auto it = count.find(i);
  return (it == count.end()) || (it->second == 0);
}

bool isPositive(const std::unordered_map<int, int> & count, int i)
{
  auto it = count.find(i);
  return (it != count.end()) && (it->second > 0);
}

// Check if count[i] != count[j]
bool haveDifferentCount(const std::unordered_map<int, int> & count, int i, int j)
{
  auto it1 = count.find(i);
  auto it2 = count.find(j);
  return (it1 == count.end() || it2 == count.end() || it1->second != it2->second);
}

bool hasCutAt(const RangeCounting & rc, int val)
{
  const auto & l = rc.limits();
  return l.end() != std::find(l.begin(), l.end(), RangeCounting::Limit(val, RangeCounting::Limit::Cut));
}

// Check that the RangeCounting is coherent with a manual, non-optimized count (where
// count[i] indicates the number of times i was added).
void checkCounting(const RangeCounting & rc, const std::unordered_map<int, int> & count, bool withSplit = false)
{
  const auto & ranges = rc.ranges(withSplit);
  if(ranges.size() == 0)
  {
    for(auto p : count)
      FAST_CHECK_EQ(p.second, 0);
  }
  else
  {
    // Check that ranges are correct
    if(withSplit)
    {
      for(const auto & r : ranges)
      {
        // The element before the range has a different count
        FAST_CHECK_UNARY(haveDifferentCount(count, r.start - 1, r.start) || hasCutAt(rc, r.start));
        // Each element in the range has a positive count
        for(int i = r.start; i < r.start + r.dim; ++i)
        {
          FAST_CHECK_UNARY(isPositive(count, i));
        }
        // The element after the range has a different count
        FAST_CHECK_UNARY(haveDifferentCount(count, r.end() - 1, r.end()) || hasCutAt(rc, r.end()));
      }
    }
    else
    {
      for(const auto & r : ranges)
      {
        // The element before the range has a count of 0
        FAST_CHECK_UNARY(isZeroOrAbsent(count, r.start - 1));
        // Each element in the range has a positive count
        for(int i = r.start; i < r.start + r.dim; ++i)
        {
          FAST_CHECK_UNARY(isPositive(count, i));
        }
        // The element after the range has a count of 0
        FAST_CHECK_UNARY(isZeroOrAbsent(count, r.end()));
      }
    }

    // Check that limits is coherent with the count
    auto d = depth(rc.limits());
    auto nonZeroc =
        std::count_if(count.begin(), count.end(), [](const std::pair<int, int> & p) { return p.second > 0; });
    auto nonZerod = std::count_if(d.begin(), d.end(), [](int i) { return i > 0; });
    FAST_CHECK_EQ(nonZeroc, nonZerod);

    int start = ranges[0].start;
    for(int i = 0; i < static_cast<int>(d.size()); ++i)
    {
      if(auto it = count.find(start + i); it != count.end())
        FAST_CHECK_EQ(d[i], it->second);
      else
        FAST_CHECK_EQ(d[i], 0);
    }
  }
}

TEST_CASE("Add/Remove")
{
  RangeCounting rc;
  std::unordered_map<int, int> count; // For ground truth

  auto add = [&rc, &count](const Range & r, bool change, int maxCount) {
    bool b = rc.add(r);
    FAST_CHECK_EQ(b, change);
    FAST_CHECK_EQ(rc.maxCount(), maxCount);
    addRange(count, r);
    checkCounting(rc, count);
  };

  auto remove = [&rc, &count](const Range & r, bool change, int maxCount) {
    bool b = rc.remove(r);
    FAST_CHECK_EQ(b, change);
    FAST_CHECK_EQ(rc.maxCount(), maxCount);
    removeRange(count, r);
    checkCounting(rc, count);
  };

  // add (3,4,5) -> (3,4,5), count [1, 1, 1]
  add({3, 3}, true, 1);

  // remove (3,4,5) -> ()
  remove({3, 3}, true, 0);

  // add (5,6) -> (5,6), count [1, 1]
  add({5, 2}, true, 1);

  // add (3,4,5) -> (3,4,5,6), count [1, 1, 2, 1]
  add({3, 3}, true, 2);

  // add (8,9) -> (3,4,5,6,8,9), count [1, 1, 2, 1,, 1, 1]
  add({8, 2}, true, 2);

  // remove (4,5,6) -> (3,5,8,9), count [1,, 1,,, 1, 1]
  remove({4, 3}, true, 1);

  // add (1,2,3,4,5,6,7) -> (1,2,3,4,5,6,7,8,9), count [1, 1, 2, 1, 2, 1, 1, 1, 1]
  add({1, 7}, true, 2);

  // remove (5,6,7,8,9) -> (1,2,3,4,5), count [1, 1, 2, 1, 1]
  remove({5, 5}, true, 2);

  // remove (1,2,3) -> (3,4,5), count [1, 1, 1]
  remove({1, 3}, true, 1);

  // remove (3,4,5) -> ()
  remove({3, 3}, true, 0);

  // add (3,4,5) -> (3,4,5), count [1 1 1]
  add({3, 3}, true, 1);

  // add (5) -> (3,4,5), count [1,1,2]
  add({5, 1}, false, 2);

  // add (0,1,2) -> (0,1,2,3,4,5), count [1,1,1,1,1,2]
  add({0, 3}, true, 2);

  // add (1,2,3) -> (0,1,2,3,4,5), count [1,2,2,2,1,2]
  add({1, 3}, false, 2);

  // remove (4,5) -> (0,1,2,3,5), count [1,2,2,2,,1]
  remove({4, 2}, true, 2);

  // remove (2,3) -> (0,1,2,3,5), count [1,2,1,1,,1]
  remove({2, 2}, false, 2);

  // remove (0,1) -> (1,2,3,5), count [1,1,1,,1]
  remove({0, 2}, true, 1);

  // add (4) -> (1,2,3,4,5), count [1,1,1,1,1]
  add({4, 1}, true, 1);

  // add () -> (1,2,3,4,5), count [1,1,1,1,1]
  add({0, 0}, false, 1);

  // add () -> (1,2,3,4,5), count [1,1,1,1,1]
  add({3, 0}, false, 1);

  // add () -> (1,2,3,4,5), count [1,1,1,1,1]
  add({6, 0}, false, 1);

  // remove (1) -> (2,3,4,5), count [1,1,1,1]
  remove({1, 1}, true, 1);

  // remove (5) -> (2,3,4), count [1,1,1]
  remove({5, 1}, true, 1);
}

TEST_CASE("Check throws")
{
  {
    RangeCounting rc;
    CHECK_THROWS(rc.remove({1, 2}));
  }
  {
    RangeCounting rc;
    rc.add({3, 4});
    CHECK_THROWS(rc.remove({1, 2})); // {1,2} not included in {3,4,5,6}
  }
  {
    RangeCounting rc;
    rc.add({3, 4});
    CHECK_THROWS(rc.remove({1, 5})); // {1,2,3,4,5} not included in {3,4,5,6}
  }
  {
    RangeCounting rc;
    rc.add({3, 4});
    CHECK_THROWS(rc.remove({3, 5})); // {3,4,5,6,7} not included in {3,4,5,6}
  }
  {
    RangeCounting rc;
    rc.add({3, 4});
    CHECK_THROWS(rc.remove({5, 3})); // {5,6,7} not included in {3,4,5,6}
  }
  {
    RangeCounting rc;
    rc.add({1, 3});
    rc.add({5, 3});
    CHECK_THROWS(rc.remove({2, 5})); // {2,3,4,5,6} not included in {1,2,3,5,6,7}
  }
}

TEST_CASE("Add/Remove Split")
{
  RangeCounting rc;
  std::unordered_map<int, int> count; // For ground truth

  auto add = [&rc, &count](const Range & r, int nbRange) {
    [[maybe_unused]] bool b = rc.add(r);
    addRange(count, r);
    checkCounting(rc, count, true);
    FAST_CHECK_EQ(rc.ranges(true).size(), nbRange);
  };

  auto remove = [&rc, &count](const Range & r, int nbRange) {
    [[maybe_unused]] bool b = rc.remove(r);
    removeRange(count, r);
    checkCounting(rc, count, true);
    FAST_CHECK_EQ(rc.ranges(true).size(), nbRange);
  };

  // add (3,4,5) -> (3,4,5), count [1, 1, 1]
  add({3, 3}, 1);

  // remove (3,4,5) -> ()
  remove({3, 3}, 0);

  // add (5,6) -> (5,6), count [1, 1]
  add({5, 2}, 1);

  // add (3,4,5) -> (3,4,5,6), count [1, 1 | 2 | 1]
  add({3, 3}, 3);

  // add (8,9) -> (3,4,5,6,8,9), count [1, 1 | 2 | 1 |, | 1, 1]
  add({8, 2}, 4);

  // remove (4,5,6) -> (3,5,8,9), count [1 |, | 1 |,, | 1, 1]
  remove({4, 3}, 3);

  // add (1,2,3,4,5,6,7) -> (1,2,3,4,5,6,7,8,9), count [1, 1 | 2 | 1 | 2 | 1, 1 | 1, 1]
  add({1, 7}, 6);

  // remove (5,6,7,8,9) -> (1,2,3,4,5), count [1, 1 | 2 | 1 | 1]
  remove({5, 5}, 4);

  // remove (1,2,3) -> (3,4,5), count [1 | 1 | 1]
  remove({1, 3}, 3);

  // remove (3,4,5) -> ()
  remove({3, 3}, 0);

  // add (3,4,5) -> (3,4,5), count [1 1 1]
  add({3, 3}, 1);

  // add (5) -> (3,4,5), count [1, 1 | 2]
  add({5, 1}, 2);

  // add (0,1,2) -> (0,1,2,3,4,5), count [1,1,1 | 1,1 | 2]
  add({0, 3}, 3);

  // add (1,2,3) -> (0,1,2,3,4,5), count [1 | 2,2 | 2 | 1 | 2]
  add({1, 3}, 5);

  // remove (4,5) -> (0,1,2,3,5), count [1 | 2,2 | 2||1]
  remove({4, 2}, 4);

  // remove (2,3) -> (0,1,2,3,5), count [1 | 2 | 1 | 1 || 1]
  remove({2, 2}, 5);

  // remove (0,1) -> (1,2,3,5), count [1 | 1 | 1 || 1]
  remove({0, 2}, 4);

  // add (4) -> (1,2,3,4,5), count [1 | 1 | 1 | 1 | 1]
  add({4, 1}, 5);

  // add () -> (1,2,3,4,5), count [1,1,1,1,1]
  add({0, 0}, 5);

  // add () -> (1,2,3,4,5), count [1,1,1,1,1]
  add({3, 0}, 5);

  // add () -> (1,2,3,4,5), count [1,1,1,1,1]
  add({6, 0}, 5);

  // remove (1) -> (2,3,4,5), count [1,1,1,1]
  remove({1, 1}, 4);

  // remove (5) -> (2,3,4), count [1,1,1]
  remove({5, 1}, 3);
}

TEST_CASE("VariableCountingVector")
{
  VariableCountingVector vc;

  VariablePtr x = Space(8).createVariable("x");
  VariablePtr y = Space(8).createVariable("y");
  VariablePtr z = Space(8).createVariable("z");

  VariablePtr x0 = x->subvariable(6, "x0", 0);
  VariablePtr x1 = x->subvariable(4, "x1", 0);
  VariablePtr x2 = x->subvariable(4, "x2", 2);
  VariablePtr x3 = x->subvariable(4, "x3", 4);
  VariablePtr x4 = x->subvariable(3, "x4", 0);
  VariablePtr x5 = x->subvariable(2, "x5", 3);
  VariablePtr x6 = x->subvariable(3, "x6", 5);

  vc.add(x);
  vc.add(y);
  vc.add(z);
  auto v = vc.variables();
  auto s = vc.simple();
  FAST_CHECK_EQ(v.numberOfVariables(), 3);
  FAST_CHECK_EQ(v.numberOfVariables(), vc.simple().size());
  FAST_CHECK_EQ(*v[0], *x);
  FAST_CHECK_EQ(*v[1], *y);
  FAST_CHECK_EQ(*v[2], *z);
  FAST_CHECK_UNARY(s[0]);
  FAST_CHECK_UNARY(s[1]);
  FAST_CHECK_UNARY(s[2]);
  FAST_CHECK_UNARY(vc.isDisjointUnion());

  vc.clear();
  v = vc.variables();
  s = vc.simple();
  FAST_CHECK_EQ(v.numberOfVariables(), 0);
  FAST_CHECK_EQ(v.numberOfVariables(), vc.simple().size());
  FAST_CHECK_UNARY(vc.isDisjointUnion());

  vc.add(y);
  vc.add(x0);
  vc.add(x1);
  vc.add(z);
  vc.add(x2);
  v = vc.variables();
  s = vc.simple();
  FAST_CHECK_EQ(v.numberOfVariables(), 3);
  FAST_CHECK_EQ(v.numberOfVariables(), vc.simple().size());
  FAST_CHECK_EQ(*v[0], *x0); // internal sorting is based on ObjectId, which is based on creation order.
  FAST_CHECK_EQ(*v[1], *y);
  FAST_CHECK_EQ(*v[2], *z);
  FAST_CHECK_UNARY(s[0]);
  FAST_CHECK_UNARY(s[1]);
  FAST_CHECK_UNARY(s[2]);
  FAST_CHECK_UNARY_FALSE(vc.isDisjointUnion());

  vc.remove(y);
  vc.remove(x1);
  v = vc.variables();
  s = vc.simple();
  FAST_CHECK_EQ(v.numberOfVariables(), 2);
  FAST_CHECK_EQ(v.numberOfVariables(), vc.simple().size());
  FAST_CHECK_EQ(*v[0], *x0);
  FAST_CHECK_EQ(*v[1], *z);
  FAST_CHECK_UNARY(s[0]);
  FAST_CHECK_UNARY(s[1]);
  FAST_CHECK_UNARY_FALSE(vc.isDisjointUnion());

  vc.remove(x0);
  v = vc.variables();
  s = vc.simple();
  FAST_CHECK_EQ(v.numberOfVariables(), 2);
  FAST_CHECK_EQ(v.numberOfVariables(), vc.simple().size());
  FAST_CHECK_EQ(*v[0], *x2);
  FAST_CHECK_EQ(*v[1], *z);
  // This is one of the case where a succession of add/remove leads to a situation where it's difficult
  // to ensure that the variable is simple, and the simple() method is conservative.
  // If a smarter method is implemented, it is fine to change this test.
  FAST_CHECK_UNARY_FALSE(s[0]);
  FAST_CHECK_UNARY(s[1]);
  FAST_CHECK_UNARY(vc.isDisjointUnion());
  vc.clear();

  vc.add(x6);
  vc.add(x4);
  v = vc.variables();
  s = vc.simple();
  FAST_CHECK_EQ(v.numberOfVariables(), 2);
  FAST_CHECK_EQ(v.numberOfVariables(), vc.simple().size());
  FAST_CHECK_EQ(*v[0], *x4); // The order here is due to internals and not mandatory
  FAST_CHECK_EQ(*v[1], *x6);
  FAST_CHECK_UNARY(s[0]);
  FAST_CHECK_UNARY(s[1]);
  FAST_CHECK_UNARY(vc.isDisjointUnion());

  vc.add(x5);
  v = vc.variables();
  s = vc.simple();
  FAST_CHECK_EQ(v.numberOfVariables(), 1);
  FAST_CHECK_EQ(v.numberOfVariables(), vc.simple().size());
  FAST_CHECK_EQ(*v[0], *x);
  FAST_CHECK_UNARY_FALSE(s[0]);
  FAST_CHECK_UNARY(vc.isDisjointUnion());

  vc.remove(x);
  vc.add(x0);
  vc.add(x1);
  v = vc.variables();
  s = vc.simple();
  FAST_CHECK_EQ(v.numberOfVariables(), 1);
  FAST_CHECK_EQ(*v[0], *x0);
  FAST_CHECK_UNARY(s[0]);
  FAST_CHECK_UNARY_FALSE(vc.isDisjointUnion());
}

TEST_CASE("VariableCountingVector split")
{
  VariableCountingVector vc(true);

  VariablePtr x = Space(8).createVariable("x");
  VariablePtr y = Space(8).createVariable("y");
  VariablePtr z = Space(8).createVariable("z");

  VariablePtr x0 = x->subvariable(6, "x0", 0);
  VariablePtr x1 = x->subvariable(4, "x1", 0);
  VariablePtr x2 = x->subvariable(4, "x2", 2);
  VariablePtr x3 = x->subvariable(4, "x3", 4);

  vc.add(x);
  vc.add(y);
  vc.add(z);
  auto v = vc.variables();
  FAST_CHECK_EQ(v.numberOfVariables(), 3);
  FAST_CHECK_EQ(*v[0], *x);
  FAST_CHECK_EQ(*v[1], *y);
  FAST_CHECK_EQ(*v[2], *z);

  vc.clear();
  v = vc.variables();
  FAST_CHECK_EQ(v.numberOfVariables(), 0);

  vc.add(y);
  vc.add(x0);
  vc.add(x1);
  vc.add(z);
  vc.add(x2);
  v = vc.variables();
  FAST_CHECK_EQ(v.numberOfVariables(), 5);
  FAST_CHECK_EQ(v[0]->subvariableRange(),
                Range(0, 2)); // internal sorting is based on ObjectId, which is based on creation order.
  FAST_CHECK_EQ(v[1]->subvariableRange(), Range(2, 2));
  FAST_CHECK_EQ(v[2]->subvariableRange(), Range(4, 2));
  FAST_CHECK_EQ(*v[3], *y);
  FAST_CHECK_EQ(*v[4], *z);

  vc.remove(x0);
  vc.remove(y);
  v = vc.variables();
  FAST_CHECK_EQ(v.numberOfVariables(), 4);
  FAST_CHECK_EQ(v[0]->subvariableRange(), Range(0, 2));
  FAST_CHECK_EQ(v[1]->subvariableRange(), Range(2, 2));
  FAST_CHECK_EQ(v[2]->subvariableRange(), Range(4, 2));
  FAST_CHECK_EQ(*v[3], *z);
  vc.clear();

  vc.add(x1);
  vc.add(x3);
  v = vc.variables();
  FAST_CHECK_EQ(v.numberOfVariables(), 2);
  FAST_CHECK_EQ(*v[0], *x1); // The order here is due to internals and not mandatory
  FAST_CHECK_EQ(*v[1], *x3);

  CHECK_THROWS(vc.simple());
}

TEST_CASE("VariableVectorPartition")
{
  VariablePtr x = Space(8).createVariable("x");
  VariablePtr y = Space(8).createVariable("y");
  VariablePtr z = Space(8).createVariable("z");

  VariablePtr x0 = x->subvariable(3, "x0", 0);
  VariablePtr x1 = x->subvariable(2, "x1", 3);
  VariablePtr x2 = x->subvariable(3, "x2", 5);
  VariablePtr y0 = y->subvariable(5, "y0", 0);
  VariablePtr y1 = y->subvariable(5, "y1", 3);
  VariablePtr z0 = z->subvariable(5, "z0", 0);
  VariablePtr z1 = z->subvariable(5, "z1", 3);

  internal::VariableCountingVector partition(true);
  partition.add(x0);
  partition.add(x1);
  partition.add(x2);
  partition.add(y0);
  partition.add(y1);
  partition.add(z0);
  partition.add(z1);

  // partition
  // <--x0--><--x1--><--x2--><---y0--->      <---z0--->
  //                               <---y1--->      <---z1--->
  // resulting in:
  // <--x0--><--x1--><--x2--><-ya-><yb><-yc-><-za-><zb><-zc->

  VariableVector var(z1, x1, y);
  std::vector<VariablePtr> u;

  //         <--x1-->        <------y ------>      <---z1--->
  // <--x0--><--x1--><--x2--><-ya-><yb><-yc-><-za-><zb><-zc->
  // {z1, x1, y} = {zb, zc, x1, ya, yb, yc}
  for(const auto & v : internal::VariableVectorPartition(var, partition))
  {
    u.push_back(v);
  }

  FAST_CHECK_EQ(u.size(), 6);
  FAST_CHECK_EQ(*u[0], *z->subvariable(2, "zb", 3));
  FAST_CHECK_EQ(*u[1], *z->subvariable(3, "zc", 5));
  FAST_CHECK_EQ(*u[2], *x1);
  FAST_CHECK_EQ(*u[3], *y->subvariable(3, "ya", 0));
  FAST_CHECK_EQ(*u[4], *y->subvariable(2, "yb", 3));
  FAST_CHECK_EQ(*u[5], *y->subvariable(3, "yc", 5));
}

TEST_CASE("Non-Euclidean variables in VariableCountingVector")
{
  VariableCountingVector vc;

  VariablePtr x = Space(21, 27, 23).createVariable("x");

  VariablePtr x1 = x->subvariable(Space(3, 3, 3), "x1", 0);
  VariablePtr x2 = x->subvariable(Space(3, 4, 3), "x2", Space(3, 3, 3));
  VariablePtr x3 = x->subvariable(Space(2, 2, 2), "x3", Space(6, 7, 6));
  VariablePtr x4 = x->subvariable(Space(4, 6, 5), "x4", Space(8, 9, 8));
  VariablePtr x5 = x->subvariable(Space(1, 1, 1), "x5", Space(12, 15, 13));
  VariablePtr x6 = x->subvariable(Space(2, 2, 2), "x6", Space(13, 16, 14));
  VariablePtr x7 = x->subvariable(Space(3, 4, 3), "x7", Space(15, 18, 16));
  VariablePtr x8 = x->subvariable(Space(2, 4, 3), "x8", Space(18, 22, 19));
  VariablePtr x9 = x->subvariable(Space(1, 1, 1), "x9", Space(20, 26, 22));

  VariablePtr x12 = x->subvariable(Space(6, 7, 6), "x12", 0);
  VariablePtr x234 = x->subvariable(Space(9, 12, 10), "x234", Space(3, 3, 3));
  VariablePtr x456 = x->subvariable(Space(7, 9, 8), "x456", Space(8, 9, 8));
  VariablePtr x678 = x->subvariable(Space(7, 10, 8), "x678", Space(13, 16, 14));
  VariablePtr x789 = x->subvariable(Space(6, 9, 7), "x789", Space(15, 18, 16));

  vc.add(x2);
  vc.add(x678);
  vc.add(x4);
  vc.add(x234);
  vc.add(x789);

  auto v = vc.variables();
  FAST_CHECK_EQ(v.numberOfVariables(), 2);

  auto dv = dot(v);
  FAST_CHECK_EQ(dv[0]->subvariableRange(), Range(3, 10));
  FAST_CHECK_EQ(dv[1]->subvariableRange(), Range(14, 9));
}
