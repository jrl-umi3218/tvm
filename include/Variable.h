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

  /** Description of a variable space, and factory for Variable.
   *
   * The space can have up to 3 different sizes, although Euclidean spaces will
   * have only one :
   * - the size of the space as a manifold,
   * - the size of the vector needed to represent one variable (point) in this
   *   space (representation space, rsize),
   * - the size of the vector needed to represent a derivative (velocity,
   *   acceleration, ...) of this variable (tangent space, tsize).
   *
   * Here are a few examples:
   * - R^n has a size, rsize and tsize of n,
   * - SO(3), the 3d rotation space, when represented by quaternions, has a
   *   size of 3, a rsize of 4 and a tsize of 3,
   * - S(2), the sphere in dimension 3 has a size of 2, and rsize and tsize of 3.
   */
  class TVM_DLLAPI Space
  {
  public:
    /** Constructor for an Euclidean space
      *
      * /param size size of the space
      */
    Space(int size);
    /** Constructor for a manifold with tsize = size
      *
      * /param size size of the space
      * /param representationSize size of the vector needed to represent a variable
      */
    Space(int size, int representationSize);
    /** Constructor for a manifold where tsize != size
      *
      * /param size size of the space
      * /param representationSize size of the vector needed to represent a variable
      * /param tangentRepresentationSize size of the vector needed to represent a derivative
      */
    Space(int size, int representationSize, int tangentRepresentationSize);

    /** Factory function to create a variable.*/
    std::unique_ptr<Variable> createVariable(const std::string& name) const;

    /** Size of the space (as a manifold) */
    int size() const;
    /** Size of the vector needed to represent a variable in this space.*/
    int rSize() const;
    /** Size of the vector needed to represent a derivative in this space.*/
    int tSize() const;
    bool isEuclidean() const;

  private:
    int mSize_;   //size of this space (as a manifold)
    int rSize_;   //size of a vector representing a point in this space
    int tSize_;   //size of a vector representing a velocity in this space
  };

  /** Get the ndiff-th time derivative of a variable
    *
    * /param var the variable to be derived
    * /param ndiff the order of the derivation
    */
  VariablePtr TVM_DLLAPI dot(VariablePtr var, int ndiff=1);


  /** Representation of a Variable, i.e. a point in a space.
    *
    * A variable can only be constructed from a Space, in which case we say it
    * is a base primitive, or from an other variable, by derivation with the
    * dot() operator.
    */
  class TVM_DLLAPI Variable
  {
  public:
    /** Return the name of the variable*/
    const std::string& name() const;
    /** Return the size of the variable, i.e. the size of the vector needed to
      * represent it.
      */
    int size() const;
    /** Return the space to which this variable is linked. It is either the
      * space the variable was obtained from (for a variable that is a
      * primitive), or the space of its primitive (for a variable which is
      * derived from an other).
      */
    const Space& space() const;
    /** Return the current value of the variable.*/
    const Eigen::VectorXd& value() const;
    /** Set the value of the variable.*/
    void value(const VectorConstRef& x);
    /** If this variable is a base primitive (i.e. built from a space), return 0.
      * Otherwise, return the number of time a base primitive variable had to be
      * derived to get to this variable.
      */
    int derivativeNumber() const;
    /** Return true if this variable is a base primitive (derivativeNumber() == 0),
      * false otherwise.
      */
    bool isBasePrimitive() const;
    /** Get the n-th primitive of the variable*/
    template <int n=1>
    VariablePtr primitive() const;
    /** Get the base primitive of this variable. Equivalent to primitive<d>()
      * where d = derivativeNumber().
      */
    VariablePtr basePrimitive() const;

    /** Get the mapping of this variable, within a vector of variables: if all
      * variables values are stacked in one vector v, the returned Range
      * specifies the segment of v corresponding to this variable.
      */
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

    /** Same as primitive<n> but without checking if n is valid.*/
    template <int n>
    VariablePtr primitiveNoCheck() const;

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
    friend class Space;
    friend VariablePtr dot(VariablePtr, int);
    friend class VariableVector;
  };

  template <int n>
  inline VariablePtr Variable::primitive() const
  {
    if (n <= derivativeNumber_)
      return primitiveNoCheck<n>();
    else
      throw std::runtime_error("This variable is not the n-th derivative of an other variable.");
  }

  template <int n>
  inline VariablePtr Variable::primitiveNoCheck() const
  {
    static_assert(n > 0, "Works only for non-negative numbers.");
    return primitive_->primitive<n - 1>();
  }

  template <>
  inline VariablePtr Variable::primitiveNoCheck<1>() const
  {
    return primitive_;
  }
}
