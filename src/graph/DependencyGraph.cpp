/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/


#include <tvm/graph/internal/DependencyGraph.h>

#include <algorithm>
#include <assert.h>
#include <numeric>
#include <stdexcept>

namespace tvm::graph::internal
{
  DependencyGraph::DisjointSet::DisjointSet(size_t n)
    : parent_(n)
    , rank_(n, 0)
  {
    // fill parents with elements from 0 to n-1, i.e. each element is its own parent
    // (all elements are in different sets)
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  size_t DependencyGraph::DisjointSet::root(size_t node)
  {
    assert(node < parent_.size());
    if (node != parent_[node])
    {
      parent_[node] = root(parent_[node]);
    }
    return parent_[node];
  }
  void DependencyGraph::DisjointSet::unify(size_t node1, size_t node2)
  {
    assert(node1 < parent_.size());
    assert(node2 < parent_.size());
    auto root1 = root(node1);
    auto root2 = root(node2);

    if (root1 == root2) return;

    if (rank_[root1] < rank_[root2])
    {
      parent_[root1] = root2;
    }
    else if (rank_[root1] > rank_[root2])
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

  bool DependencyGraph::addEdge(size_t from, size_t to)
  {
    assert(from < roots_.size());
    assert(to < roots_.size());

    auto p = edges_.insert({ from, to });

    if (p.second)
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

  std::vector<std::vector<size_t>> DependencyGraph::order() const
  {
    size_t n = roots_.size();
    std::vector<size_t> order;
    order.reserve(n);
    std::vector<bool> visited(n, false);
    std::vector<bool> stack(n, false);
    DisjointSet components(n);

    bool has_root = false;
    for (size_t i = 0; i < visited.size(); ++i)
    {
      if (roots_[i])
      {
        orderUtil(i, order, visited, stack, components);
        has_root = true;
      }
    }

    if (!has_root && visited.size() != 0)
    {
      throw std::logic_error("[DependencyGraph::Order] Try to order a non-empty graph with no root. It contains at least one cycle.");
    }

    //At this point, we have all the nodes ordered in order, but without the notion
    //of components. On the other hand, parent describes a collection of trees,
    //each one corresponding to a component. We need to merge those data.
    std::vector<std::vector<size_t>> orderedGroups;

    //First, we create a map associating a group number to a root.
    std::vector<size_t> map(n);
    for (size_t i = 0; i < n; ++i)
    {
      if (components.isRoot(i))
      {
        map[i] = orderedGroups.size();
        orderedGroups.push_back({});
      }
    }

    //Second, we populate the groups in an ordered way
    for (auto i : order)
    {
      orderedGroups[map[components.root(i)]].push_back(i);
    }

    return orderedGroups;
  }

  size_t DependencyGraph::size() const
  {
    return roots_.size();
  }

  const std::set<std::pair<size_t, size_t>>& DependencyGraph::edges() const
  {
    return edges_;
  }

  void DependencyGraph::clear()
  {
    roots_.clear();
    children_.clear();
    edges_.clear();
  }


  std::pair<std::vector<std::vector<size_t>>, DependencyGraph>  DependencyGraph::reduce() const
  {
    size_t n = roots_.size();
    std::vector<int> disc(n, -1);
    std::vector<int> low(n, -1);
    std::vector<bool> stackMember(n, false);
    std::stack<size_t> st;
    int time = 0;

    std::vector<std::vector<size_t>> ret;

    //Call the recursive helper function to find SCC.
    for (size_t i = 0; i < n; i++)
      if (disc[i] == -1)
        SCCUtil(ret, i, disc, low, st, stackMember, time);

    //Now we need to build the reduced graph.
    DependencyGraph g;

    //  First, we build a map between a vertex and its SCC.
    //  For each SCC we add a node in the reduced graph.
    std::vector<size_t> vert2SCC(n);
    for (size_t i = 0; i < ret.size(); ++i)
    {
      for (auto j : ret[i])
      {
        vert2SCC[j] = i;
      }
      g.addNode();
    }

    //  Second, we add the edges
    for (const auto& p : edges_)
    {
      auto f = vert2SCC[p.first];
      auto t = vert2SCC[p.second];
      if (f != t)  // we ignore edges between elements of a same SCC
      {
        g.addEdge(f, t);
      }
    }

    return { ret, g };
  }

  void DependencyGraph::SCCUtil(std::vector<std::vector<size_t>>& ret,
    size_t u, std::vector<int>& disc, std::vector<int>& low,
    std::stack<size_t>& st, std::vector<bool>& stackMember, int& time) const
  {
    assert(time >= 0);

    // Initialize discovery time and low value
    disc[u] = low[u] = ++time;
    st.push(u);
    stackMember[u] = true;

    // Go through all vertices adjacent to this
    for (auto v : children_[u])
    {
      // If v is not visited yet, then recur for it
      if (disc[v] == -1)
      {
        SCCUtil(ret, v, disc, low, st, stackMember, time);

        // Check if the subtree rooted with 'v' has a
        // connection to one of the ancestors of 'u'
        low[u] = std::min(low[u], low[v]);
      }
      else if (stackMember[v])
      {
        // Update low value of 'u' only if 'v' is still in stack
        // (i.e. it's a back edge, not cross edge).
        low[u] = std::min(low[u], disc[v]);
      }
    }

    // head node found, pop the stack and push an SCC
    if (low[u] == disc[u])
    {
      ret.push_back({});
      while (st.top() != u)
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

  void DependencyGraph::orderUtil(size_t v, std::vector<size_t>& order,
    std::vector<bool>& visited, std::vector<bool>& stack,
    DisjointSet& components) const
  {
    if (!visited[v])
    {
      visited[v] = true;
      stack[v] = true;

      for (auto i : children_[v])
      {
        components.unify(v, i);
        if (!visited[i])
        {
          orderUtil(i, order, visited, stack, components);
        }
        else if (stack[i])
        {
          throw std::logic_error("[DependencyGraph::orderUtil] The graph contains a cycle");
        }
      }
    }
    stack[v] = false;
    order.push_back(v);
  }
}