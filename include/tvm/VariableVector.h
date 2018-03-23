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
    /** Remove a variable from the vector.
      *
      * \param v the variable to be removed
      *
      * \returns True if the variable was removed, false otherwise
      */
    bool remove(const Variable& v);

    /** Sum of the sizes of all the variables.*/
    int size() const;
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
    /** Check if this vector contains variable v or not.*/
    bool contains(const Variable& v) const;

    /** A timestamp, used internally to determine if a mapping needs to be
      * recomputed or not.
      */
    int stamp() const;

  private:
    void getNewStamp() const;

    static int counter;

    mutable int stamp_;
    int size_;
    std::vector<VariablePtr> variables_;
    /** This set is a helper to quickly test the presence of a variable without
      * iterating through the whole vector.
      *
      * FIXME: is it faster though, given that we will never have a lot of variables?
      */
    std::set<const Variable*> variableSet_;

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

}  // namespace tvm
