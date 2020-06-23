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

#pragma once

#include <tvm/api.h>

#include <vector>
#include <set>
#include <stack>

namespace tvm::graph::internal
{
  /** An oriented graph to represent dependencies between substitutions.*/
  class TVM_DLLAPI DependencyGraph
  {
  public:
    /** Create a node and return its id.*/
    size_t addNode();
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
    /** Return the list of node groups. Each group corresponds to a connected
      * component of the graph seen as non-oriented. Within each group the
      * nodes are sorted in reversed topological order.
      */
    std::vector<std::vector<size_t>> order() const;
    /** Number of vertices*/
    size_t size() const;
    /** List of edges*/
    const std::set<std::pair<size_t, size_t>>& edges() const;

  private:
    /** A recursive function that finds and prints strongly connected
      * components (SCC) using DFS traversal
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
    void SCCUtil(std::vector<std::vector<size_t>>& ret,
      size_t u, std::vector<int>& disc, std::vector<int>& low,
      std::stack<size_t>& st, std::vector<bool>& stackMember, int& time) const;

    /** A recursive function used for topological ordering. It also detects
      * connected components of the graph.
      * \param v the vertex to process
      * \param order the vertices in reversed topological order
      * \param visited visited[i] is true if and only if vertex i was visited
      * \param stack stack[i] is true if and only if vertex i is among the
      * vertices being processed.
      * \parent a description of a tree collection used for detecting the
      * connected components. For the first call to orderUtil, it must have a
      * size equal to the number of vertices and be initialized so that
      * parent[i] = i.
      * \rank data used for the detection of the components. For the first
      * call to orderUtil, it must have a size equal to the number of vertices
      * and be initialized to 0.
      */
    void orderUtil(size_t v, std::vector<size_t>& order,
      std::vector<bool>& visited, std::vector<bool>& stack,
      std::vector<size_t>& parent, std::vector<size_t>& rank) const;


    std::vector<bool> roots_;                       //roots_[i] is true iff the i-th node has no incoming edges
    std::vector<std::vector<size_t>> children_;     //children_[i] lists all the childs of node i
    std::set<std::pair<size_t, size_t>> edges_;     //a list of edges (from, to)

  };
}