#include <tvm/hint/internal/Substitutions.h>
#include <tvm/Variable.h>
#include <tvm/constraint/abstract/LinearConstraint.h>

#include <algorithm>



namespace
{
  /** Find the root of the tree the element \p node is in.
    * The structure of the tree is given by \p parent.
    * The function also perform a path compression, in that all node discovered
    * along the way are made to point directly to the root.
    *
    * This function is part of an implementation of the so-called disjoint-set
    * data structure.
    */
  size_t root(size_t node, std::vector<size_t>& parent)
  {
    assert(node < parent.size());
    if (node != parent[node])
    {
      parent[node] = root(parent[node], parent);
    }
    return parent[node];
  }

  /** Unify the tree \p node1 is in with the tree \p node2 is in.
    * This is done by having the root of one of the trees point at the root of
    * the other.
    * The structure of the trees is given by \p parent.
    * Chosing wich root points to which is done based on the rank of both roots,
    * where the rank is a heuristic number approximating the tree depth. The
    * root with the smallest rank point to the other, in which case the rank of
    * the other remain unchanged. If both roots have the same rank, the second
    * point to the first and the rank of the first one is incremented by one.
    * The ranks are stored in the aptly named \p rank.
    *
    * This function is part of an implementation of the so-called disjoint-set
    * data structure.
    */
  void unify(size_t node1, size_t node2, std::vector<size_t>& parent, std::vector<size_t>& rank)
  {
    assert(rank.size() == parent.size());
    assert(node1 < parent.size());
    assert(node2 < parent.size());
    auto root1 = root(node1, parent);
    auto root2 = root(node2, parent);

    if (root1 == root2) return;

    if (rank[root1] < rank[root2])
    {
      parent[root1] = root2;
    }
    else if (rank[root1] > rank[root2])
    {
      parent[root2] = root1;
    }
    else
    {
      parent[root2] = root1;
      rank[root1]++;
    }
  }

  /** Initialize the vectors describing a disjoint-set data structure.
    *
    * This function is part of an implementation of the so-called disjoint-set
    * data structure.
    */
  void initialize(std::vector<size_t>& parent, std::vector<size_t>& rank)
  {
    assert(rank.size() == parent.size());
    for (size_t i = 0; i < rank.size(); ++i)
    {
      rank[i] = 0;
      parent[i] = i;
    }
  }
}

namespace tvm
{

namespace hint
{

namespace internal
{
  /** Return true if \p s is using variables substituted by \p t.*/
  bool dependsOn(const Substitution& s, const Substitution& t)
  {
    for (const auto& x : t.variables())
    {
      for (const auto& c : s.constraints())
      {
        if (c->variables().contains(*x))
        {
          return true;
        }
      }
    }
    return false;
  }


  void tvm::hint::internal::Substitutions::add(const Substitution& s)
  {
    auto i = dependencies_.addNode();
    assert(i == substitutions_.size());
    substitutions_.push_back(s);

    for (size_t j=0; j<substitutions_.size(); ++j)
    {
      if (dependsOn(substitutions_[j], s))
      {
        dependencies_.addEdge(j, i);
      }
      if (dependsOn(s, substitutions_[j]))
      {
        dependencies_.addEdge(i, j);
      }
    }
  }

  void Substitutions::finalize()
  {
    //Detect interdependent substitutions (this corresponds to strongly connected
    //components of the dependency graph).
    DependencyGraph g;
    std::vector<std::vector<size_t>> scc;
    std::tie(scc,g) = dependencies_.reduce();

    //Compute the groups of substitutions and the order to carry out the substitutions
    //in each group. Indices in orderedGroups are relative to scc.
    auto orderedGroups = g.order();

    //We create a unit for each group
    units_.clear();
    for (const auto& g : orderedGroups)
    {
      units_.emplace_back(substitutions_, scc, g);
    }

    //Retrieve all the variables, functions and constraints
    variables_.clear();
    varSubstitutions_.clear();
    additionalConstraints_.clear();
    for (const auto& u : units_)
    {
      const auto& x = u.variables();
      const auto& f = u.variableSubstitutions();
      const auto& z = u.additionalVariables();
      const auto& c = u.additionalConstraints();
      variables_.insert(variables_.end(), x.begin(), x.end());
      varSubstitutions_.insert(varSubstitutions_.end(), f.begin(), f.end());
      for (auto& zi : z)
      {
        if (zi->size() > 0)
        {
          additionalVariables_.push_back(zi);
        }
      }
      for (auto& ci : c)
      {
        if (ci->size() > 0)
        {
          additionalConstraints_.push_back(ci);
        }
      }
    }
  }

  void Substitutions::updateSubstitutions()
  {
    for (auto& u : units_)
    {
      u.update();
    }
  }

  void Substitutions::updateVariableValues() const
  {
    for (int i = 0; i < variables_.size(); ++i)
    {
      varSubstitutions_[i]->updateValue();
      variables_[i]->value(varSubstitutions_[i]->value());
    }
  }

  const std::vector<VariablePtr>& Substitutions::variables() const
  {
    return variables_;
  }

  const std::vector<std::shared_ptr<function::BasicLinearFunction>>& Substitutions::variableSubstitutions() const
  {
    return varSubstitutions_;
  }

  const std::vector<VariablePtr>& Substitutions::additionalVariables() const
  {
    return additionalVariables_;
  }

  const std::vector<std::shared_ptr<constraint::BasicLinearConstraint>>& Substitutions::additionalConstraints() const
  {
    return additionalConstraints_;
  }

  size_t Substitutions::DependencyGraph::addNode()
  {
    roots_.push_back(true);
    children_.push_back({});
    return roots_.size() - 1;
  }

  bool Substitutions::DependencyGraph::addEdge(size_t from, size_t to)
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

  std::vector<std::vector<size_t>> Substitutions::DependencyGraph::order() const
  {
    size_t n = roots_.size();
    std::vector<size_t> order;
    order.reserve(n);
    std::vector<bool> visited(n, false);
    std::vector<bool> stack(n, false);
    std::vector<size_t> parent(n);
    std::vector<size_t> rank(n);
    initialize(parent, rank);

    bool has_root = false;
    for (size_t i = 0; i < visited.size(); ++i)
    {
      if (roots_[i])
      {
        orderUtil(i, order, visited, stack, parent, rank);
        has_root = true;
      }
    }

    if (!has_root && visited.size() != 0)
    {
      throw std::logic_error("Try to order a non-empty graph with no root. It contains at least one cycle.");
    }

    //At this point, we have all the nodes ordered in order, but without the notion
    //of components. On the other hand, parent describes a collection of trees,
    //each one corresponding to a component. We need to merge those data.
    std::vector<std::vector<size_t>> orderedGroups;

    //First, we create a map associating a group number to a root.
    std::vector<size_t> map(n);
    for (size_t i=0; i<n; ++i)
    {
      if (parent[i] == i)
      {
        map[i] = orderedGroups.size();
        orderedGroups.push_back({});
      }
    }

    //Second, we populate the groups in an ordered way
    for (auto i : order)
    {
      orderedGroups[map[root(i, parent)]].push_back(i);
    }

    return orderedGroups;
  }

  size_t Substitutions::DependencyGraph::size() const
  {
    return roots_.size();
  }

  const std::set<std::pair<size_t, size_t>>& Substitutions::DependencyGraph::edges() const
  {
    return edges_;
  }


  std::pair<std::vector<std::vector<size_t>>, Substitutions::DependencyGraph>  Substitutions::DependencyGraph::reduce() const
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
    for (size_t i= 0; i<ret.size(); ++i)
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

  void Substitutions::DependencyGraph::SCCUtil(std::vector<std::vector<size_t>>& ret,
                                               size_t u, std::vector<int>& disc, std::vector<int>& low,
                                               std::stack<size_t>& st, std::vector<bool>& stackMember, int& time) const
  {
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

    // head node found, pop the stack and print an SCC
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

  void Substitutions::DependencyGraph::orderUtil(size_t v, std::vector<size_t>& order,
          std::vector<bool>& visited, std::vector<bool>& stack,
          std::vector<size_t>& parent, std::vector<size_t>& rank) const
  {
    if (!visited[v])
    {
      visited[v] = true;
      stack[v] = true;

      for (auto i : children_[v])
      {
        unify(v, i, parent, rank);
        if (!visited[i]) 
        {
          orderUtil(i, order, visited, stack, parent, rank); 
        }
        else if (stack[i])
        {
          throw std::logic_error("The graph contains a cycle");
        }
      }
    }
    stack[v] = false;
    order.push_back(v);
  }


} // internal

} // hint

} // tvm
