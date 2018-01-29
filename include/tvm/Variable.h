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

#include <tvm/Range.h>
#include <tvm/Space.h>

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
  class TVM_DLLAPI Variable : public std::enable_shared_from_this<Variable>
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
    return primitive_;
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

/** Helper to set the value of a variable v to val. */
inline tvm::VariablePtr& operator<<(tvm::VariablePtr& v, const tvm::VectorConstRef& val)
{
  v->value(val);
  return v;
}

/** Helper to initialize a variable like an Eigen vector, with a coma separated list.*/
inline Eigen::CommaInitializer<Eigen::VectorXd> operator<<(tvm::VariablePtr& v, double d)
{
  return *v.get() << d;
}

/** Helper to initialize a variable like an Eigen vector, with a coma separated list.*/
template<typename Derived>
inline Eigen::CommaInitializer<Eigen::VectorXd> operator<<(tvm::VariablePtr& v, const Eigen::DenseBase<Derived>& other)
{
  return *v.get() << other;
}