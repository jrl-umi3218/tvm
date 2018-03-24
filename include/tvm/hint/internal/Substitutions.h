#pragma once

/* Copyright 2017-2018 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/hint/Substitution.h>
#include <tvm/hint/internal/SubstitutionUnit.h>

#include <vector>
#include <set>
#include <stack>

namespace tvm
{

namespace hint
{

namespace internal
{
  /** A set of substitutions*/
  class TVM_DLLAPI Substitutions
  {
  public:
    /** Add substitution \p s*/
    void add(const Substitution& s);

    /** Get the vector of all substitutions, as added.
      * Note that it is not necessarily the vector of substitutions actually
      * used, as it might be needed to group substitutions (when a group of
      * substitutions depends on each other variables).
      */
    const std::vector<Substitution>& substitutions() const;

    /** Return \p true if \p c is used in one of the substitutions*/
    bool uses(LinearConstraintPtr c) const;

    /** Compute all the data needed for the substitutions.
      * Needs to be called after all the call to \p add, and before the calls to
      * \p variables, \pvariableSubstitutions and \p additionalConstraints.
      */
    void finalize();

    /** Update the data for the substitutions*/
    void updateSubstitutions();

    /** Update the value of the substituted variables according to the values of
      * the non-substitued ones.*/
    void updateVariableValues() const;

    /** All variables x in the substitutions*/
    const std::vector<VariablePtr>& variables() const;
    /** The linear functions x = f(y,z) corresponding to the variables*/
    const std::vector<std::shared_ptr<function::BasicLinearFunction>>& variableSubstitutions() const;
    /** The additional nullspace variables z*/
    const std::vector<VariablePtr>& additionalVariables() const;
    /** The remaining constraints on y and z*/
    const std::vector<std::shared_ptr<constraint::BasicLinearConstraint>>& additionalConstraints() const;
    /** The variables y*/
    const std::vector<VariablePtr>& otherVariables() const;
    /** If \p x is one of the substituted variables, returns the variables it is
      * replaced by. Otherwise, return \p x
      */
    VariableVector substitute(const VariablePtr& x) const;

    /** An oriented graph to represent dependencies between substitutions.*/
  private:
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
        * \param visited visited[i] is true iff vertex i was visited
        * \param stack stack[i] is true iff vertex i is among the vertices being
        * processed.
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

    /** The substitutions, as added to the objects*/
    std::vector<Substitution> substitutions_;

    /** Dependency graph between the substitutions. There is an edge from i to j
      * if the substitutions_[i] relies on substitutions_[j].
      */
    DependencyGraph dependencies_;

    /** Group of dependent substitutions*/
    std::vector<SubstitutionUnit> units_;

    /** The variables substituted (x).*/
    std::vector<VariablePtr> variables_;

    /** The substitution functions linked to the variables, i.e
      * variables_[i].value() is given by varSubstitutions_[i].value().
      */
    std::vector<std::shared_ptr<function::BasicLinearFunction>> varSubstitutions_;

    /** Nullspace variables (z) used in the substitutions*/
    std::vector<VariablePtr> additionalVariables_;

    /** Other variables (y), i.e. the variables present in the constraints used
      * for the substitutions but not substituted.
      */
    std::vector<VariablePtr> otherVariables_;

    /** The additionnal constraints to add to the problem*/
    std::vector<std::shared_ptr<constraint::BasicLinearConstraint>> additionalConstraints_;

    friend class SubstitutionTest;
  };

}

}

}
