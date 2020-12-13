/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>

#include <tvm/graph/internal/AbstractNode.h>
#include <tvm/graph/internal/DependencyGraph.h>

namespace tvm
{

namespace graph
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
  void add(std::shared_ptr<internal::Inputs> inputs);

  /** Update the plan */
  void update();

  /** Execute the plan */
  inline void execute() const { plan_.execute(); }

  /** Clear the object.*/
  void clear();

protected:
  /** A call is formed by the combination of a Node and id */
  struct Call
  {
    internal::AbstractNode * node;
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

    /** Clear the plan*/
    void clear();

    /** Execute the plan */
    inline void execute() const
    {
      for(auto & c : plan_)
      {
        c();
      }
    }

  private:
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
  std::vector<int> addOutput(abstract::Outputs * source, int output);

  /** Add a Call to the graph
   *
   * It has no effect if the Call is already in the graph.
   *
   * \returns The Call id
   */
  int addCall(Call c);

  /** Store inputs added to the graph */
  std::vector<std::shared_ptr<internal::Inputs>> inputs_;
  /** Call ids */
  std::map<Call, int, CompareCall> callId_;
  /** Id to Call */
  std::vector<Call> calls_;
  /** Dependency graph*/
  internal::DependencyGraph dependencyGraph_;

  /** Used to avoid duplicate entries in the graph */
  std::map<std::intptr_t, std::map<int, std::vector<int>>> visited_;

  /** Store the call plan */
  Plan plan_;
};

} // namespace graph

} // namespace tvm
