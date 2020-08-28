/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/VariableVector.h>
#include <tvm/scheme/internal/ProblemDefinitionEvent.h>

#include <queue>
#include <vector>

namespace tvm::scheme::internal
{

  /** Base class to be derived by each resolution scheme to hold all of the
    * temporary memories and states related to the resolution of a problem.
    * The rationale is that a resolution scheme is stateless, and when given
    * for the first time a problem to solve, creates a ProblemComputationData
    * that it gives to the problem and retrieves at each further resolution.
    * A problem computation data further act as a queue of events changing the
    * problem definition.
    */
  class TVM_DLLAPI ProblemComputationData
  {
  public:
    virtual ~ProblemComputationData() = default;

    int solverId() const;

    /** Add a variable if not already present. Return true if the variable was
      * effectively added. The class counts how many time a variable was added.
      */
    bool addVariable(VariablePtr var);
    void addVariable(const VariableVector& vars);
    /** Effectively remove a variable if as many calls were made to this method
      * as to addVariable, for the given variable. Return true if the variable
      * was really removed.
      */
    bool removeVariable(Variable* v);
    void removeVariable(const VariableVector& vars);
    const VariableVector& variables() const;

    /** Set the value of the variables to that of the solution. */
    void setVariablesToSolution();

    /** Adding an event changing the problem definition, that needs to be handled
      * by the resolution scheme. Events are stored in a queue (FIFO).
      */
    void addEvent(const ProblemDefinitionEvent& e);
    /** Read and remove the first event in the queue. Need the queue to be non-empty.*/
    ProblemDefinitionEvent popEvent();
    /** Is the queue not empty?*/
    bool hasEvents() const;

  protected:
    ProblemComputationData(int solverId);
    ProblemComputationData() = delete;

    /** Need to put in x the solution of the computation. */
    virtual void setVariablesToSolution_(VariableVector& x) = 0;

    /** The problem variable*/
    VariableVector x_;
    /** Count on the number of time a variable was added to x_.*/
    std::vector<int> varCount_;

    /** Problem definition events*/
    std::queue<ProblemDefinitionEvent> events_;

  private:
    int solverId_;
  };

  inline int ProblemComputationData::solverId() const
  {
    return solverId_;
  }

  inline bool ProblemComputationData::addVariable(VariablePtr var)
  {
    assert(x_.numberOfVariables() == varCount_.size());
    size_t i = x_.addAndGetIndex(var);
    if (i >= varCount_.size())
    {
      varCount_.push_back(1);
      return true;
    }
    else
    {
      ++varCount_[i];
      return false;
    }
  }

  inline void ProblemComputationData::addVariable(const VariableVector& vars)
  {
    for (const auto& v : vars.variables())
      addVariable(v);
  }

  inline bool ProblemComputationData::removeVariable(Variable* v)
  {
    int i = x_.indexOf(*v);
    assert(i >= 0 && i < static_cast<int>(varCount_.size()));

    --varCount_[i];
    if (varCount_[i] == 0)
    {
      x_.remove(i);
      return true;
    }
    else
    {
      return false;
    }
  }

  inline void ProblemComputationData::removeVariable(const VariableVector& vars)
  {
    for (const auto& v : vars.variables())
      removeVariable(v.get());
  }

  inline const VariableVector& ProblemComputationData::variables() const
  {
    return x_;
  }

  inline void ProblemComputationData::setVariablesToSolution()
  {
    setVariablesToSolution_(x_);
  }

  inline void ProblemComputationData::addEvent(const ProblemDefinitionEvent& e)
  {
    events_.push(e);
  }

  [[nodiscard]] inline ProblemDefinitionEvent ProblemComputationData::popEvent()
  {
    assert(hasEvents());
    auto e = events_.front();
    events_.pop();
    return e;
  }

  inline bool ProblemComputationData::hasEvents() const
  {
    return !events_.empty();
  }

  inline ProblemComputationData::ProblemComputationData(int solverId)
    : solverId_(solverId)
  {
  }
}
