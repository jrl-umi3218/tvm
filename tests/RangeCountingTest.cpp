/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/internal/RangeCounting.h>

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
    if(it->lower_)
      ++depth;
    else
      --depth;

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

// Check that the RangeCounting is coherent with a manual, non-optimized count (where
// count[i] indicates the number of times i was added).
void checkCounting(const RangeCounting & rc, const std::unordered_map<int, int> & count)
{
  const auto & ranges = rc.ranges();
  if(ranges.size() == 0)
  {
    for(auto p : count)
      FAST_CHECK_EQ(p.second, 0);
  }
  else
  {
    // Check that ranges are correct
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
      FAST_CHECK_UNARY(isZeroOrAbsent(count, r.start + r.dim));
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

  auto add = [&rc, &count](const Range & r) {
    rc.add(r);
    addRange(count, r);
    checkCounting(rc, count);
  };

  auto remove = [&rc, &count](const Range & r) {
    rc.remove(r);
    removeRange(count, r);
    checkCounting(rc, count);
  };

  // add (3,4,5) -> (3,4,5)
  add({3, 3});

  // remove (3,4,5) -> ()
  remove({3, 3});

  // add (5,6) -> (3,4,5,6)
  add({5, 2});

  // add (3,4,5) -> (3,4,5)
  add({3, 3});

  // add (8,9) -> (3,4,5,6,8,9)
  add({8, 2});

  // remove (4,5,6) -> (3,5,8,9)
  remove({4, 3});

  // add (1,2,3,4,5,6,7) -> (1,2,3,4,5,6,7,8,9)
  add({1, 7});

  // remove (5,6,7,8,9) -> (1,2,3,4,5)
  remove({5, 5});

  // remove (1,2,3) -> (3,4,5)
  remove({1, 3});

  // remove (3,4,5) -> ()
  remove({3, 3});
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