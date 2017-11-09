#pragma once

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <tvm/api.h>
#include "defs.h"

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
    * contained variables will be computed and cached. For repeatidly querying
    * the mapping of those variable w.r.t the same VariableVector, this is the
    * fastest option. However it will be slow if querying alternatively
    * mapping w.r.t different VariableVector on the same variable or set of
    * variables.
    *
    * A given variable can only appear once in a vector.
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
      * /param v the variable to be added
      * /param mergeDuplicate if true, attempting to add a variable that is
      * already in the vector will be ignored. If false, it will raise an exception.
      */
    void add(VariablePtr v, bool mergeDuplicate = false);
    /** Same as add(VariablePtr, bool), but for adding a vector of variables.*/
    void add(const std::vector<VariablePtr>& variables, bool mergeDuplicate = false);
    /** Remove a variable from the vector.
      *
      * /param v the variable to be removed
      * /param ignoreAbsence if true, attempting to remove a variable that is
      * not present in the vector will be ignored, If false, it will raise an
      * exception.
      */
    void remove(const Variable& v, bool ignoreAbsence = false);

    /** Sum of the sizes of all the variables.*/
    int size() const;
    /** Number of variables*/
    int numberOfVariables() const;
    /** Elementwise access*/
    const VariablePtr operator[](int i) const;
    /** whole vector access*/
    const std::vector<VariablePtr>& variables() const;

    /** Get the concatenation of all variables' value.
      *
      * /warning this operation requires a memory allocation at each call.
      */
    Eigen::VectorXd value() const;
    /** Set the value of all variables from a concatenated vector*/
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
  };
}