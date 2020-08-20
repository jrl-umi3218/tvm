/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
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
#include <tvm/defs.h>
#include <tvm/VariableVector.h>
#include <tvm/scheme/internal/ProblemDefinitionEvent.h>

#include <queue>
#include <vector>

namespace tvm
{

namespace scheme
{

namespace internal
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

    void addVariable(VariablePtr var);
    void addVariable(const VariableVector& vars);
    void removeVariable(Variable* v);
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

    /** Problem definition events*/
    std::queue<ProblemDefinitionEvent> events_;

  private:
    int solverId_;
  };

  inline int ProblemComputationData::solverId() const
  {
    return solverId_;
  }

  inline void ProblemComputationData::addVariable(VariablePtr var)
  {
    x_.add(var);
  }

  inline void ProblemComputationData::addVariable(const VariableVector& vars)
  {
    x_.add(vars);
  }

  inline void ProblemComputationData::removeVariable(Variable* v)
  {
    //we don't raise an exception is the variable is not there, as we merge
    //identical variables when we add them.
    x_.remove(*v);
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

  inline ProblemDefinitionEvent ProblemComputationData::popEvent()
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

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
