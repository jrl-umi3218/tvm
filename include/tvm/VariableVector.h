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
#include <tvm/internal/MatrixWithProperties.h>

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace tvm
{
  /** A vector of variables, with some useful manipulation/analysis functions.
    *
    * One of the main use of this class is to determine variable mapping, i.e.
    * given a vector aggregating all variables, which section of this vector
    * would correspond to a given variable.
    * There is two approaches for that: either build a map with
    * computeMappingMap or use the method Variable::getMappingIn. The latter
    * uses a cache in Variable in a way that if one invoke Variable::getMappingIn
    * on any variable contained in a VariableVector, the mapping of all other
    * contained variables will be computed and cached. For repeatedly querying
    * the mapping of those variable w.r.t the same VariableVector, this is the
    * fastest option. However it will be slow if querying alternatively
    * mapping w.r.t different VariableVector on the same variable or set of
    * variables.
    *
    * A given variable can only appear once in a vector. Variables appear in the
    * order they were added, ignoring duplicates.
    *
    * FIXME would it make sense to derive from std::vector<std::shared_ptr<Variable>> ?
    */
  class TVM_DLLAPI VariableVector
  {
  public:
    VariableVector();
    VariableVector(const std::vector<VariablePtr>& variables);
    VariableVector(std::initializer_list<VariablePtr> variables);

    /** Add a variable to the vector.
      *
      * \param v the variable to be added
      *
      * \returns True if the variable was added, false otherwise
      */
    bool add(VariablePtr v);
    /** Same as add(VariablePtr), but for adding a vector of variables.*/
    void add(const std::vector<VariablePtr>& variables);
    /** Same as add(VariablePtr), but for adding a vector of variables.*/
    void add(const VariableVector& variables);
    /** Remove a variable from the vector.
      *
      * \param v the variable to be removed
      *
      * \returns True if the variable was removed, false otherwise
      */
    bool remove(const Variable& v);

    /** Sum of the sizes of all the variables.*/
    int totalSize() const;
    /** Number of variables*/
    int numberOfVariables() const;
    /** Elementwise access*/
    const VariablePtr operator[](int i) const;
    /** whole vector access*/
    const std::vector<VariablePtr>& variables() const;

    /** Get the concatenation of all variables' value, in the order of the
      * variables as given by variables().
      */
    const Eigen::VectorXd& value() const;
    /** Set the value of all variables from a concatenated vector
      *
      * \param val The concatenated value of all the variables, in the order of
      * the variables as given by variables().
      */
    void value(const VectorConstRef& val);

    /** Compute the mapping for all variables in this vector. The result is
      * stored in each variable and can be queried by Variable::getMappingIn.
      */
    void computeMapping() const;

    /** Compute the mapping for every variabe and return it.*/
    std::map<const Variable*, Range> computeMappingMap() const;
    /** Check if this vector contains variable \p v or not. */
    bool contains(const Variable& v) const;

    /** Find the index of variable \p v in the vector. Returns -1 if \p v is not
      * present.
      */
    int indexOf(const Variable& v) const;

    /** A timestamp, used internally to determine if a mapping needs to be
      * recomputed or not.
      */
    int stamp() const;

    /** Iterator to the first variable. This enable to use VariableVector
      * directly in range-based for loops
      */
    std::vector<VariablePtr>::const_iterator begin() const;
    /** Iterator past the last variable. This enable to use VariableVector
      * directly in range-based for loops
      */
    std::vector<VariablePtr>::const_iterator end() const;

  private:
    void getNewStamp() const;

    static int counter;

    mutable int stamp_;
    int size_;
    std::vector<VariablePtr> variables_;

    mutable Eigen::VectorXd value_;
  };

  /** Get the vector of ndiff-th time derivatives of the variables of the input
    * vector.
    *
    * \param var the variable to be derived
    * \param ndiff the order of the derivation
    *
    * \warning This recreates a vector from scratch each time
    */
  VariableVector TVM_DLLAPI dot(const VariableVector& vars, int ndiff=1);


  inline std::vector<VariablePtr>::const_iterator tvm::VariableVector::begin() const
  {
    return variables_.begin();
  }

  inline std::vector<VariablePtr>::const_iterator tvm::VariableVector::end() const
  {
    return variables_.end();
  }

}  // namespace tvm
