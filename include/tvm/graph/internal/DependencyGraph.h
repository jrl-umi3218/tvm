/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>

#include <cstdint>
#include <set>
#include <stack>
#include <vector>

namespace tvm::graph::internal
{
/** An oriented graph to represent dependencies between nodes.
 *
 * An edge (f, t) represents a dependency from \p f on \p t.
 */
class TVM_DLLAPI DependencyGraph
{
public:
  /** Create a node and return its id.*/
  size_t addNode();
  /** Remove a node and all its associated edges */
  void removeNode(size_t id);
  /** Add an edge from node with id \p from to node with id \p to.
   * Return \p true if this indeed creates a new edge, \p false if this
   * edge was already present.
   */
  bool addEdge(size_t from, size_t to);
  /** Return the reduced graph of this graph as a pair (v,g)
   * v is a list of groups of vertices, each group corresponding to a
   * strongly connected component.
   * g is the direct acyclic graph between these groups
   */
  std::pair<std::vector<std::vector<size_t>>, DependencyGraph> reduce() const;
  /** Return the list of nodes in reversed topological order.*/
  std::vector<size_t> order() const;
  /** Return the list of node groups. Each group corresponds to a connected
   * component of the graph seen as non-oriented. Within each group the
   * nodes are sorted in reversed topological order.
   */
  std::vector<std::vector<size_t>> groupedOrder() const;
  /** Number of vertices*/
  size_t size() const;
  /** List of edges*/
  const std::set<std::pair<size_t, size_t>> & edges() const;
  /** Clear the graph.*/
  void clear();

private:
  /** This class implements a minimalist disjoint set data structure.
   * See https://en.wikipedia.org/wiki/Disjoint-set_data_structure
   *
   * Given a number n, there are n elements with implicit id from 0 to n-1.
   * Each element point to a parent and has a rank. This defines in
   * particular trees, where a root point to itself.
   * The rank is a heuristic on the depth at which a node is in its tree.
   */
  class DisjointSet
  {
  public:
    /** Initialize for 0 elements*/
    DisjointSet() = default;

    /** Initialize for n elements*/
    DisjointSet(size_t n);

    /** Resize to n elements*/
    void resize(size_t n);

    /** Find the root of the tree the element \p node is in.
     */
    size_t root(size_t node) const;

    /** Unify the tree \p node1 is in with the tree \p node2 is in.
     * This is done by having the root of one of the trees point at the root of
     * the other.
     * Choosing which root points to which is done based on the rank of both roots,
     * where the rank is a heuristic number approximating the tree depth. The
     * root with the smallest rank point to the other, in which case the rank of
     * the other remain unchanged. If both roots have the same rank, the second
     * point to the first and the rank of the first one is incremented by one.
     * The ranks are stored in the aptly named \p rank.
     */
    void unify(size_t node1, size_t node2);

    /** Check if node i is a root.*/
    bool isRoot(size_t i) const;

  private:
    /** Find the root of the tree the element \p node is in.
     * The function also perform a path compression, in that all node discovered
     * along the way are made to point directly to the root.
     */
    size_t rootCompr(size_t node);

    std::vector<size_t> parent_; // parent_[i] is the id of the parent of element i
    std::vector<size_t> rank_;   // rank of each element.
  };

  /** A recursive function that finds and returns strongly connected components
   * (SCC) using DFS traversal.
   * \param ret a vector of SCC
   * \param u the vertex to visit
   * \param disc discovery time of the vertices (should be -1 if not
   * discovered yet)
   * \param low earliest visited vertex (the vertex with minimum discovery
   * time) that can be reached from subtree rooted with current vertex
   * \param st stack to store all connected ancestors
   * \param stackMember array for checking faster if a node is in the stack
   * \param time a counter increased at each entry in the function. Should
   * be non-negative.
   *
   * Adapted from
   * https://www.geeksforgeeks.org/tarjan-algorithm-find-strongly-connected-components/
   */
  void SCCUtil(std::vector<std::vector<size_t>> & ret,
               size_t u,
               std::vector<int> & disc,
               std::vector<int> & low,
               std::stack<size_t> & st,
               std::vector<uint8_t> & stackMember,
               int & time) const;

  /** An intermediate function for computing the topological order.*/
  std::pair<std::vector<size_t>, DisjointSet> order_() const;

  /** A recursive function used for topological ordering. It also detects
   * connected components of the graph.
   * \param v the vertex to process
   * \param order the vertices in reversed topological order
   * \param visited visited[i] is true if and only if vertex i was visited
   * \param stack stack[i] is true if and only if vertex i is among the
   * vertices being processed.
   * \param components A disjoint-set data structure representing the
   * connected components
   */
  void orderUtil(size_t v,
                 std::vector<size_t> & order,
                 std::vector<uint8_t> & visited,
                 std::vector<uint8_t> & stack,
                 DisjointSet & components) const;

  std::vector<uint8_t> roots_;                // roots_[i] is true iff the i-th node has no incoming edges
  std::vector<std::vector<size_t>> children_; // children_[i] lists all the children of node i
  std::set<std::pair<size_t, size_t>> edges_; // a list of edges (from, to)
};
} // namespace tvm::graph::internal
