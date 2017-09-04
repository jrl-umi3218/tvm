#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
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

#include "data/AbstractNode.h"

namespace tvm
{

/** A Direct Acyclic Graph representing the dependencies between update
 * calls where an edge (from, to) means that from depends on to.
 *
 * The calls present in the graph depend on the inputs of the Inputs
 * objects added to the graph.
 */
class TVM_DLLAPI CallGraph
{
public:
  /** Add a given Inputs object to the graph */
  void add(std::shared_ptr<data::Inputs> inputs);

  /** Update the plan */
  void update();

  /** Execute the plan */
  inline void execute() const
  {
    plan_.execute();
  }
protected:
  /** A call is formed by the combination of a Node and and id */
  struct Call
  {
    std::shared_ptr<data::AbstractNode> node;
    int id;
    inline void operator()() const { node->update(id); }
  };

  /** Similar to a specialization of std::less for Call */
  struct CompareCall
  {
    bool operator()(const Call & c1, const Call & c2) const
    {
      return (c1.node < c2.node) || (c1.node == c2.node && c1.id < c2.id);
    }
  };

  /** An execution plan built from the graph */
  struct Plan
  {
    /** Build the plan from the CallGraph */
    void build(const CallGraph & graph);

    /** Execute the plan */
    inline void execute() const
    {
      for(auto & c : plan_)
      {
        c();
      }
    }
  private:
    void recursiveBuild(const CallGraph & graph, size_t v,
                        std::vector<size_t> & order,
                        std::vector<bool> & visited,
                        std::vector<bool> & stack);
    /** The calls in the correct call order */
    std::vector<Call> plan_;
  };

protected:
  /** Add a given source and its output to the graph
   *
   * It has no effect if the source and its output are already in the
   * graph.
   *
   * \returns The list of dependencies for the given source.
   */
  std::vector<int> addOutput(const std::shared_ptr<data::Outputs> & source,
                             int output);

  /** Add a Call to the graph
   *
   * It has no effect if the Call is already in the graph.
   *
   * \returns The Call id
   */
  int addCall(Call c);

  /** Add an edge to the graph */
  void addEdge(int from, int to);

  /** Call ids */
  std::map<Call, int, CompareCall> callId_;
  /** Id to Call */
  std::vector<Call> calls_;
  /** Dependencies id of each call (edges) */
  std::vector<std::vector<int>> dependencies_;
  /** root_[id] is true if id is a root of the graph */
  std::vector<bool> root_;

  /** Used to avoid duplicate entries in the graph */
  std::map<std::intptr_t, std::map<int, std::vector<int> > > visited_;

  /** Store the call plan */
  Plan plan_;
};

} // namespace tvm
