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

    /** Set the value of the variables. \a val must be in the same order as the
      * variables.
      */
    void setSolution(const VectorConstRef& val);

  protected:
    ProblemComputationData(int solverId);
    ProblemComputationData() = default;

    /** The problem variable*/
    VariableVector x_;

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

  inline void ProblemComputationData::setSolution(const VectorConstRef & val)
  {
    x_.value(val);
  }

  inline ProblemComputationData::ProblemComputationData(int solverId)
    : solverId_(solverId)
  {
  }

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
