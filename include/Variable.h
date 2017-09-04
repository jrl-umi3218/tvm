#pragma once

#include <Eigen/Core>

#include <memory>
#include <vector>
#include <string>

#include "defs.h"
#include "tvm/api.h"

namespace tvm
{
  class Variable;
  class VariableVector;
  
  class Range
  {
  public:
    Range() : start(0), dim(0) {}
    Range(int s, int d) : start(s), dim(d) {}
    int start;
    int dim;
  };


  class TVM_DLLAPI Space
  {
  public:
    Space(int size);
    Space(int size, int representationSize);
    Space(int size, int representationSize, int tangentRepresentationSize);

    std::unique_ptr<Variable> createVariable(const std::string& name) const;

    int size() const;
    int rSize() const;
    int tSize() const;

  private:
    int mSize_;   //size of this space (as a manifold)
    int rSize_;   //size of a vector representing a point in this space
    int tSize_;   //size of a vector representing a velocity in this space
  };

  VariablePtr TVM_DLLAPI dot(VariablePtr var, int ndiff=1);

  class TVM_DLLAPI Variable
  {
  public:
    const std::string& name() const;
    int size() const;
    const Space& space() const;
    const Eigen::VectorXd& value() const;
    void value(const VectorConstRef& x);
    int derivativeNumber() const;
    bool isBasePrimitive() const;
    VariablePtr primitive() const;
    VariablePtr basePrimitive() const;

    Range getMappingIn(const VariableVector& variables) const;

  protected:

  private:
    struct MappingHelper
    {
      int start;
      int stamp;
    };

    /** Constructor for a new variable */
    Variable(const Space& s, const std::string& name);

    /** Constructor for the derivative of var */
    Variable(VariablePtr var);


    /** name */
    std::string name_;

    /** data of the space from which the variable was created */
    Space space_;

    /** Value of the variable */
    Eigen::VectorXd value_;

    /** Number of derivation since the base primitive. 0 a variable which is 
      * not a derivative.
      */
    int derivativeNumber_;

    /** If the variable is the time derivative of another one, primitive_ is a 
      * reference to the latter, otherwise it is uninitialized.
      */
    VariablePtr primitive_;

    /** If the variable has a time derivative, keep a pointer on it */
    std::weak_ptr<Variable> derivative_;

    /** A helper structure for mapping purpose*/
    mutable MappingHelper mappingHelper_;

    /** friendship declaration */
    friend class TVM_DLLAPI Space;
    friend VariablePtr TVM_DLLAPI dot(VariablePtr, int);
    friend class TVM_DLLAPI VariableVector;
  };
}
