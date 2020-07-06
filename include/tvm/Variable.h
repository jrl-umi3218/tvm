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

#include <tvm/Range.h>
#include <tvm/Space.h>

#include <tvm/internal/ObjWithId.h>

#include <Eigen/Core>

#include <memory>
#include <vector>
#include <string>

namespace tvm
{
  class Variable;
  class VariableVector;

  /** Get the ndiff-th time derivative of a variable
    *
    * \param var the variable to be derived
    * \param ndiff the order of the derivation
    */
  VariablePtr TVM_DLLAPI dot(VariablePtr var, int ndiff=1);


  /** Representation of a Variable, i.e. a point in a space.
    *
    * A variable can only be constructed from a Space, in which case we say it
    * is a base primitive, or from an other variable, by derivation with the
    * dot() operator.
    */
  class TVM_DLLAPI Variable : public tvm::internal::ObjWithId, public std::enable_shared_from_this<Variable>
  {
  public:
    /** Copying variables is illicit. To get a different variable with the same
      * caracteristics, see \ref duplicate.
      */
    Variable(const Variable&) = delete;
    /** Copying variables is illicit. To get a different variable with the same
      * caracteristics, see \ref duplicate.
      */
    Variable& operator=(const Variable&) = delete;
    /** Create a variable based on the same space, with the same derivative
      * number and value.
      * \param name the name of the new variable. If the default value is kept,
      * a ' will be appened to the current name. If the variable being
      * duplicated is not a base variable, the name is given to the base
      * variable.
      */
    VariablePtr duplicate(const std::string& name = "") const;
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
    /** Check if the variable lives in a Euclidean space.
      * Pay attention that for a variable v, v.isEuclidean() is not necessarily
      * equal to v.space().isEuclidean(): if v is a derivative of a variable linked
      * to a non-Euclidean space, it lives in the tangent space which is Euclidean.
      */
    bool isEuclidean() const;
    /** Return the current value of the variable.*/
    const Eigen::VectorXd& value() const;
    /** Set the value of the variable.*/
    void value(const VectorConstRef& x);
    /** If this variable is a base primitive (i.e. built from a space), return 0.
      * Otherwise, return the number of time a base primitive variable had to be
      * derived to get to this variable.
      */
    int derivativeNumber() const;
    /** Check if this variable is a derivative of \p v
      * This is in a strict sense: v.isDerivativeOf(v) si false.
      */
    bool isDerivativeOf(const Variable& v) const;
    /** Check if this variable is a primitive of \p v
      * This is in a strict sense: v.isPrimitiveOf(v) si false.
      */
    bool isPrimitiveOf(const Variable& v) const;
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

    /** Helper to initialize a variable like an Eigen vector.*/
    Eigen::CommaInitializer<Eigen::VectorXd> operator<<(double d);

    /** Helper to initialize a variable like an Eigen vector.*/
    template<typename Derived>
    Eigen::CommaInitializer<Eigen::VectorXd> operator<<(const Eigen::DenseBase<Derived>& other);

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
    Variable(Variable* var);

    /** Same as primitive<n> but without checking if n is valid.*/
    template <int n>
    VariablePtr primitiveNoCheck() const;

    /** Name */
    std::string name_;

    /** Data of the space from which the variable was created */
    Space space_;

    /** Value of the variable */
    Eigen::VectorXd value_;

    /** Number of derivation since the base primitive. 0 a variable which is
      * not a derivative.
      */
    int derivativeNumber_;

    /** If the variable is the time derivative of another one, primitive_ is a
      * reference to the latter, otherwise it is a nullptr.
      */
    Variable* primitive_;

    /** If the variable has a time derivative, keep a pointer on it */
    std::unique_ptr<Variable> derivative_;

    /** A helper structure for mapping purpose*/
    mutable MappingHelper mappingHelper_;

    /** friendship declaration */
    friend class Space;
    friend VariablePtr TVM_DLLAPI dot(VariablePtr, int);
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
    if (derivativeNumber_ > 1)
      return { basePrimitive(), primitive_ };
    else
      return primitive_->shared_from_this();
  }

  inline Eigen::CommaInitializer<Eigen::VectorXd> Variable::operator<<(double d)
  {
    return { value_,d };
  }

  template<typename Derived>
  inline Eigen::CommaInitializer<Eigen::VectorXd> Variable::operator<<(const Eigen::DenseBase<Derived>& other)
  {
    return { value_, other };
  }
}  // namespace tvm

/** Helper to initialize a variable like an Eigen vector, with a coma separated list.*/
inline Eigen::CommaInitializer<Eigen::VectorXd> operator<<(tvm::VariablePtr& v, double d)
{
  return *v.get() << d;
}

/** Helper to initialize a variable like an Eigen vector, with a coma separated list.*/
inline Eigen::CommaInitializer<Eigen::VectorXd> operator<<(tvm::VariablePtr&& v, double d)
{
  return *v.get() << d;
}

/** Helper to initialize a variable like an Eigen vector, with a coma separated list.*/
template<typename Derived>
inline Eigen::CommaInitializer<Eigen::VectorXd> operator<<(tvm::VariablePtr& v, const Eigen::DenseBase<Derived>& other)
{
  return *v.get() << other;
}

/** Helper to initialize a variable like an Eigen vector, with a coma separated list.*/
template<typename Derived>
inline Eigen::CommaInitializer<Eigen::VectorXd> operator<<(tvm::VariablePtr&& v, const Eigen::DenseBase<Derived>& other)
{
  return *v.get() << other;
}