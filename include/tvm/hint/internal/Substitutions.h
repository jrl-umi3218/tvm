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
#include <tvm/graph/internal/DependencyGraph.h>
#include <tvm/hint/Substitution.h>
#include <tvm/hint/internal/SubstitutionUnit.h>

#include <vector>

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
      * \p variables, \p variableSubstitutions and \p additionalConstraints.
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

  private:
    /** The substitutions, as added to the objects*/
    std::vector<Substitution> substitutions_;

    /** Dependency graph between the substitutions. There is an edge from i to j
      * if the substitutions_[i] relies on substitutions_[j].
      */
    tvm::graph::internal::DependencyGraph dependencies_;

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
