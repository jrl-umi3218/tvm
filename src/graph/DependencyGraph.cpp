/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/graph/internal/DependencyGraph.h>

#include <algorithm>
#include <assert.h>
#include <numeric>
#include <stdexcept>

namespace tvm::graph::internal
{
DependencyGraph::DisjointSet::DisjointSet(size_t n) : parent_(n), rank_(n, 0)
{
  // fill parents with elements from 0 to n-1, i.e. each element is its own parent
  // (all elements are in different sets)
  std::iota(parent_.begin(), parent_.end(), 0);
}

void DependencyGraph::DisjointSet::resize(size_t n)
{
  parent_.resize(n);
  std::iota(parent_.begin(), parent_.end(), 0);
  rank_.resize(n);
  std::fill_n(rank_.begin(), n, 0);
}

size_t DependencyGraph::DisjointSet::root(size_t node) const
{
  assert(node < parent_.size());
  if(node != parent_[node])
  {
    return root(parent_[node]);
  }
  return parent_[node];
}

size_t DependencyGraph::DisjointSet::rootCompr(size_t node)
{
  assert(node < parent_.size());
  if(node != parent_[node])
  {
    parent_[node] = rootCompr(parent_[node]);
  }
  return parent_[node];
}

void DependencyGraph::DisjointSet::unify(size_t node1, size_t node2)
{
  assert(node1 < parent_.size());
  assert(node2 < parent_.size());
  auto root1 = rootCompr(node1);
  auto root2 = rootCompr(node2);

  if(root1 == root2)
    return;

  if(rank_[root1] < rank_[root2])
  {
    parent_[root1] = root2;
  }
  else if(rank_[root1] > rank_[root2])
  {
    parent_[root2] = root1;
  }
  else
  {
    parent_[root2] = root1;
    rank_[root1]++;
  }
}

bool DependencyGraph::DisjointSet::isRoot(size_t i) const
{
  assert(i < parent_.size());
  return parent_[i] == i;
}

size_t DependencyGraph::addNode()
{
  roots_.push_back(true);
  children_.push_back({});
  return roots_.size() - 1;
}

void DependencyGraph::removeNode(size_t id)
{
  using Edge = std::pair<size_t, size_t>;
  assert(id < roots_.size());
  // Remove all edges from/to {id}
  for(auto it = edges_.begin(); it != edges_.end();)
  {
    const Edge & e = *it;
    if(e.first == id || e.second == id)
    {
      it = edges_.erase(it);
    }
    else
    {
      ++it;
    }
  }
  // We check if the children of {id} can be promoted to roots_
  // i.e. if they have no more edges pointing to them
  for(auto c : children_[id])
  {
    if(std::find_if(edges_.begin(), edges_.end(), [&](const Edge & e) { return e.second == c; }) == edges_.end())
    {
      roots_[c] = true;
    }
  }
  for(size_t i = 0; i < roots_.size(); ++i)
  {
    if(i == id)
    {
      continue;
    }
    // Do two things at once:
    // 1. Decrement the edge id for every edge which is above id
    // 2. Remove all edges matching the removed edge
    children_[i].erase(std::remove_if(children_[i].begin(), children_[i].end(),
                                      [&](size_t & j) {
                                        if(j > id)
                                        {
                                          j -= 1;
                                          return false;
                                        }
                                        return j == id;
                                      }),
                       children_[i].end());
  }
  roots_.erase(roots_.begin() + id);
  children_.erase(children_.begin() + id);
}

bool DependencyGraph::addEdge(size_t from, size_t to)
{
  assert(from < roots_.size());
  assert(to < roots_.size());

  auto p = edges_.insert({from, to});

  if(p.second)
  {
    children_[from].push_back(to);
    roots_[to] = false;
    return true;
  }
  else
  {
    return false;
  }
}

std::vector<size_t> DependencyGraph::order() const { return order_().first; }

std::vector<std::vector<size_t>> DependencyGraph::groupedOrder() const
{
  size_t n = roots_.size();
  const auto & [order, components] = order_();

  // At this point, we have all the nodes ordered in order, but without the notion
  // of components. On the other hand, parent describes a collection of trees,
  // each one corresponding to a component. We need to merge those data.
  std::vector<std::vector<size_t>> orderedGroups;

  // First, we create a map associating a group number to a root.
  std::vector<size_t> map(n);
  for(size_t i = 0; i < n; ++i)
  {
    if(components.isRoot(i))
    {
      map[i] = orderedGroups.size();
      orderedGroups.push_back({});
    }
  }

  // Second, we populate the groups in an ordered way
  for(auto i : order)
  {
    orderedGroups[map[components.root(i)]].push_back(i);
  }

  return orderedGroups;
}

size_t DependencyGraph::size() const { return roots_.size(); }

const std::set<std::pair<size_t, size_t>> & DependencyGraph::edges() const { return edges_; }

void DependencyGraph::clear()
{
  roots_.clear();
  children_.clear();
  edges_.clear();
}

std::pair<std::vector<std::vector<size_t>>, DependencyGraph> DependencyGraph::reduce() const
{
  size_t n = roots_.size();
  std::vector<int> disc(n, -1);
  std::vector<int> low(n, -1);
  std::vector<uint8_t> stackMember(n, false);
  std::stack<size_t> st;
  int time = 0;

  std::vector<std::vector<size_t>> ret;

  // Call the recursive helper function to find SCC.
  for(size_t i = 0; i < n; i++)
    if(disc[i] == -1)
      SCCUtil(ret, i, disc, low, st, stackMember, time);

  // Now we need to build the reduced graph.
  DependencyGraph g;

  //  First, we build a map between a vertex and its SCC.
  //  For each SCC we add a node in the reduced graph.
  std::vector<size_t> vert2SCC(n);
  for(size_t i = 0; i < ret.size(); ++i)
  {
    for(auto j : ret[i])
    {
      vert2SCC[j] = i;
    }
    g.addNode();
  }

  //  Second, we add the edges
  for(const auto & p : edges_)
  {
    auto f = vert2SCC[p.first];
    auto t = vert2SCC[p.second];
    if(f != t) // we ignore edges between elements of a same SCC
    {
      g.addEdge(f, t);
    }
  }

  return {ret, g};
}

void DependencyGraph::SCCUtil(std::vector<std::vector<size_t>> & ret,
                              size_t u,
                              std::vector<int> & disc,
                              std::vector<int> & low,
                              std::stack<size_t> & st,
                              std::vector<uint8_t> & stackMember,
                              int & time) const
{
  assert(time >= 0);

  // Initialize discovery time and low value
  disc[u] = low[u] = ++time;
  st.push(u);
  stackMember[u] = true;

  // Go through all vertices adjacent to this
  for(auto v : children_[u])
  {
    // If v is not visited yet, then recur for it
    if(disc[v] == -1)
    {
      SCCUtil(ret, v, disc, low, st, stackMember, time);

      // Check if the subtree rooted with 'v' has a
      // connection to one of the ancestors of 'u'
      low[u] = std::min(low[u], low[v]);
    }
    else if(stackMember[v])
    {
      // Update low value of 'u' only if 'v' is still in stack
      // (i.e. it's a back edge, not cross edge).
      low[u] = std::min(low[u], disc[v]);
    }
  }

  // head node found, pop the stack and push an SCC
  if(low[u] == disc[u])
  {
    ret.push_back({});
    while(st.top() != u)
    {
      ret.back().push_back(st.top());
      stackMember[st.top()] = false;
      st.pop();
    }
    ret.back().push_back(st.top());
    stackMember[st.top()] = false;
    st.pop();
  }
}

std::pair<std::vector<size_t>, DependencyGraph::DisjointSet> DependencyGraph::order_() const
{
  size_t n = roots_.size();
  std::pair<std::vector<size_t>, DependencyGraph::DisjointSet> ret;
  ret.first.reserve(n);
  ret.second.resize(n);
  std::vector<uint8_t> visited(n, false);
  std::vector<uint8_t> stack(n, false);

  bool has_root = false;
  for(size_t i = 0; i < visited.size(); ++i)
  {
    if(roots_[i])
    {
      orderUtil(i, ret.first, visited, stack, ret.second);
      has_root = true;
    }
  }

  if(!has_root && visited.size() != 0)
  {
    throw std::logic_error(
        "[DependencyGraph::Order] Try to order a non-empty graph with no root. It contains at least one cycle.");
  }

  return ret;
}

void DependencyGraph::orderUtil(size_t v,
                                std::vector<size_t> & order,
                                std::vector<uint8_t> & visited,
                                std::vector<uint8_t> & stack,
                                DisjointSet & components) const
{
  if(!visited[v])
  {
    visited[v] = true;
    stack[v] = true;

    for(auto i : children_[v])
    {
      components.unify(v, i);
      if(!visited[i])
      {
        orderUtil(i, order, visited, stack, components);
      }
      else if(stack[i])
      {
        throw std::logic_error("[DependencyGraph::orderUtil] The graph contains a cycle");
      }
    }
  }
  stack[v] = false;
  order.push_back(v);
}
} // namespace tvm::graph::internal
