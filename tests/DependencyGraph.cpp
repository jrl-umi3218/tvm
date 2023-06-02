/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/graph/internal/DependencyGraph.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

struct compVec
{
  bool operator()(const std::vector<size_t> & u, const std::vector<size_t> & v) const
  {
    if(u.size() < v.size())
      return true;
    else if(v.size() < u.size())
      return false;
    else
      return std::set<size_t>(u.begin(), u.end()) < std::set<size_t>(v.begin(), v.end());
  }
};

bool equal(const std::vector<std::vector<size_t>> & u, const std::vector<std::vector<size_t>> & v)
{
  if(u.size() == v.size())
  {
    std::set<std::vector<size_t>, compVec> su(u.begin(), u.end());
    std::set<std::vector<size_t>, compVec> sv(v.begin(), v.end());
    // using ==, <=, ... between two sets seems to always use ==, <=, ... on the
    // elements of the set, irrespective of the comparator given to the set. We
    // thus relies on std::lexicographical_compare with our custom comparator
    return !std::lexicographical_compare(su.begin(), su.end(), sv.begin(), sv.end(), compVec())
           && !std::lexicographical_compare(sv.begin(), sv.end(), su.begin(), su.end(), compVec());
  }
  else
    return false;
}

TEST_CASE("SCC test")
{
  tvm::graph::internal::DependencyGraph g;
  g.addNode();
  g.addNode();     //  0---+
  g.addNode();     //  |   |
  g.addNode();     //  |   v
  g.addNode();     //  |   3
  g.addNode();     //  |   |
  g.addEdge(0, 3); //  1<--+
  g.addEdge(3, 1); //  |
  g.addEdge(1, 0); //  +-->4
  g.addEdge(1, 2); //  |
  g.addEdge(1, 4); //  v
  g.addEdge(2, 5); //  2<->5
  g.addEdge(5, 2);

  auto r = g.reduce();
  const auto & e = r.second.edges();

  FAST_CHECK_UNARY(equal(r.first, {{0, 1, 3}, {4}, {2, 5}}));
  FAST_CHECK_EQ(e.size(), 2);
  // we have two edges, one from the SCC with size 3 to the SCC with size 1,
  // one from the SCC with size 3 to the SCC with size 2
  auto it1 = e.begin();
  auto it2 = e.begin();
  ++it2;
  FAST_CHECK_EQ(r.first[it1->first].size(), 3);
  FAST_CHECK_EQ(r.first[it2->first].size(), 3);
  auto s1 = r.first[it1->second].size();
  auto s2 = r.first[it2->second].size();
  FAST_CHECK_UNARY((s1 == 1 && s2 == 2) || (s1 == 2 && s2 == 1));
}

TEST_CASE("Order test")
{
  tvm::graph::internal::DependencyGraph g;
  g.addNode();
  g.addNode();
  g.addNode();     //  +-->1---+
  g.addNode();     //  |       |
  g.addNode();     //  0       |
  g.addNode();     //  |       v
  g.addNode();     //  +-->2-->4
  g.addNode();     //          ^
  g.addNode();     //          |
  g.addNode();     //  8-->9---+
  g.addEdge(0, 1); //
  g.addEdge(0, 2); //  +-->5
  g.addEdge(1, 2); //  |   |
  g.addEdge(1, 4); //  3   |
  g.addEdge(2, 4); //  |   v
  g.addEdge(3, 5); //  +-->6
  g.addEdge(3, 6); //
  g.addEdge(5, 6); //  7
  g.addEdge(8, 9);
  g.addEdge(9, 4);

  auto r = g.order();
  auto find = [&r](int i) { return std::find(r.begin(), r.end(), i); };
  FAST_CHECK_LT(find(4), find(1));
  FAST_CHECK_LT(find(4), find(2));
  FAST_CHECK_LT(find(4), find(9));
  FAST_CHECK_LT(find(1), find(0));
  FAST_CHECK_LT(find(2), find(0));
  FAST_CHECK_LT(find(9), find(8));
  FAST_CHECK_LT(find(6), find(5));
  FAST_CHECK_LT(find(6), find(3));
  FAST_CHECK_LT(find(5), find(3));

  auto rg = g.groupedOrder();
  std::set<std::vector<size_t>> vg = {{4, 2, 1, 0, 9, 8}, {6, 5, 3}, {7}};
  FAST_CHECK_EQ(std::set<std::vector<size_t>>(rg.begin(), rg.end()), vg);
}
