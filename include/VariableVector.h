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
    * There is two approaches for that. Either build a map with 
    * computeMappingMap or use the method Variable::getMappingIn. The latter
    * uses a cache in Variable in a way that if one invoke Variable::getMappingIn
    * on any variable contained in a VariableVector, the mapping of all other
    * contained variables will be computed and cached. For repeatidly querying
    * the mapping of those variable w.r.t the same VariableVector, this is the
    * fastest option. However it will be slow if querying alternatively
    * mapping w.r.t different VariableVector on the same variable or set of
    * variables.
    *
    * FIXME would it make sense to derive from std::vector<std::shared_ptr<Variable>> ?
    */
  class TVM_DLLAPI VariableVector
  {
  public:
    VariableVector();
    VariableVector(const std::vector<VariablePtr>& variables);
    VariableVector(std::initializer_list<VariablePtr> variables);

    void add(VariablePtr v, bool mergeDuplicate = false);
    void add(const std::vector<VariablePtr>& variables, bool mergeDuplicate = false);
    void remove(const Variable& v, bool ignoreAbsence = false);

    /** Sum of the sizes of all the variables.*/
    int size() const;
    /** Number of variables*/
    int numberOfVariables() const;
    /** Elementwise access*/
    const VariablePtr operator[](int i) const;
    /** whole vector access*/
    const std::vector<VariablePtr>& variables() const;

    /** read/write
      *
      * be careful that the read operation needs to allocate memory
      */
    Eigen::VectorXd value() const;
    void value(const VectorConstRef& val);

    //mapping related methods
    void computeMapping() const;
    std::map<const Variable*, Range> computeMappingMap() const;
    bool contains(const Variable& v) const;
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