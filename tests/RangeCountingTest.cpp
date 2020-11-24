/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Variable.h>

#include <tvm/internal/RangeCounting.h>
#include <tvm/internal/VariableCountingVector.h>

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

  auto add = [&rc, &count](const Range & r, bool change) {
    bool b = rc.add(r);
    FAST_CHECK_EQ(b, change);
    addRange(count, r);
    checkCounting(rc, count);
  };

  auto remove = [&rc, &count](const Range & r, bool change) {
    bool b = rc.remove(r);
    FAST_CHECK_EQ(b, change);
    removeRange(count, r);
    checkCounting(rc, count);
  };

  // add (3,4,5) -> (3,4,5), count [1, 1, 1]
  add({3, 3}, true);

  // remove (3,4,5) -> ()
  remove({3, 3}, true);

  // add (5,6) -> (5,6), count [1, 1]
  add({5, 2}, true);

  // add (3,4,5) -> (3,4,5,6), count [1, 1, 2, 1]
  add({3, 3}, true);

  // add (8,9) -> (3,4,5,6,8,9), count [1, 1, 2, 1,, 1, 1]
  add({8, 2}, true);

  // remove (4,5,6) -> (3,5,8,9), count [1,, 1,,, 1, 1]
  remove({4, 3}, true);

  // add (1,2,3,4,5,6,7) -> (1,2,3,4,5,6,7,8,9), count [1, 1, 2, 1, 2, 1, 1, 1, 1]
  add({1, 7}, true);

  // remove (5,6,7,8,9) -> (1,2,3,4,5), count [1, 1, 2, 1, 1]
  remove({5, 5}, true);

  // remove (1,2,3) -> (3,4,5), count [1, 1, 1]
  remove({1, 3}, true);

  // remove (3,4,5) -> ()
  remove({3, 3}, true);

  // add (3,4,5) -> (3,4,5), count [1 1 1]
  add({3, 3}, true);

  // add (5) -> (3,4,5), count [1,1,2]
  add({5, 1}, false);

  // add (0,1,2) -> (0,1,2,3,4,5), count [1,1,1,1,1,2]
  add({0, 3}, true);

  // add (1,2,3) -> (0,1,2,3,4,5), count [1,2,2,2,1,2]
  add({1, 3}, false);

  // remove (4,5) -> (0,1,2,3,5), count [1,2,2,2,,1]
  remove({4, 2}, true);

  // remove (2,3) -> (0,1,2,3,5), count [1,2,1,1,,1]
  remove({2, 2}, false);

  // remove (0,1) -> (1,2,3,5), count [1,1,1,,1]
  remove({0, 2}, true);

  // add (4) -> (1,2,3,4,5), count [1,1,1,1,1]
  add({4, 1}, true);

  // add () -> (1,2,3,4,5), count [1,1,1,1,1]
  add({0, 0}, false);

  // add () -> (1,2,3,4,5), count [1,1,1,1,1]
  add({3, 0}, false);

  // add () -> (1,2,3,4,5), count [1,1,1,1,1]
  add({6, 0}, false);

  // remove (1) -> (2,3,4,5), count [1,1,1,1]
  remove({1, 1}, true);

  // remove (5) -> (2,3,4), count [1,1,1]
  remove({5, 1}, true);
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
    bool b = rc.add(r);
    addRange(count, r);
    checkCounting(rc, count, true);
    FAST_CHECK_EQ(rc.ranges(true).size(), nbRange);
  };

  auto remove = [&rc, &count](const Range & r, int nbRange) {
    bool b = rc.remove(r);
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

  vc.clear();
  v = vc.variables();
  s = vc.simple();
  FAST_CHECK_EQ(v.numberOfVariables(), 0);
  FAST_CHECK_EQ(v.numberOfVariables(), vc.simple().size());

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

  vc.add(x5);
  v = vc.variables();
  s = vc.simple();
  FAST_CHECK_EQ(v.numberOfVariables(), 1);
  FAST_CHECK_EQ(v.numberOfVariables(), vc.simple().size());
  FAST_CHECK_EQ(*v[0], *x);
  FAST_CHECK_UNARY_FALSE(s[0]);

  vc.remove(x);
  vc.add(x0);
  vc.add(x1);
  v = vc.variables();
  s = vc.simple();
  FAST_CHECK_EQ(v.numberOfVariables(), 1);
  FAST_CHECK_EQ(*v[0], *x0);
  FAST_CHECK_UNARY(s[0]);
}