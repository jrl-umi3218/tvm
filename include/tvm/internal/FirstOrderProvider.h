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

#include <tvm/defs.h>
#include <tvm/VariableVector.h>
#include <tvm/graph/abstract/Node.h>
#include <tvm/internal/MatrixWithProperties.h>

#include <Eigen/Core>

#include <algorithm>
#include <map>
#include <vector>

namespace tvm
{

namespace internal
{

  /** Describes an entity that can provide a value and its jacobian*/
  class TVM_DLLAPI FirstOrderProvider : public graph::abstract::Node<FirstOrderProvider>
  {
  public:
    SET_OUTPUTS(FirstOrderProvider, Value, Jacobian)

    /** Note: by default, these methods return the cached value.
    * However, they are virtual in case the user might want to bypass the cache.
    * This would be typically the case if he/she wants to directly return the
    * output of another method, e.g. return the jacobian of an other Function.
    */
    virtual const Eigen::VectorXd& value() const;
    virtual const MatrixWithProperties& jacobian(const Variable& x) const;

    /** Linearity w.r.t x*/
    bool linearIn(const Variable& x) const;

    /** Return the output size m*/
    int size() const;

    /** Return the variables*/
    const std::vector<VariablePtr>& variables() const;

  protected:
    /** Constructor
      * /param m the output size of the function/constraint, i.e. the size of
      * the value (or equivalently the row size of the jacobians).
      */
    FirstOrderProvider(int m);

    /** Resize all cache members corresponding to active outputs.
      *
      * This can be overriden in case you do not need all of the default
      * mechanism (typically if you will not use part of the cache).
      * If you override to perform additional operations, do not forget to
      * call this base version in the derived classes.
      */
    virtual void resizeCache();

    /** Sub-methods of resizeCache to be used by derived classes that need
      * this level of granularity.
      */
    void resizeValueCache();
    void resizeJacobianCache();

    /** Add a variable. Cache is automatically updated.
      * \param v The variable to add/remove
      * \param linear Specify that the entity is depending linearly on the
      * variable or not.
      */
    void addVariable(VariablePtr v, bool linear);
    /** Remove a variable. Cache is automatically updated. */
    void removeVariable(VariablePtr v);

    /** To be overriden by derived classes that need to react to
      * the addition of a variable. Called at the end of addVariable();
      */
    virtual void addVariable_(VariablePtr);
    virtual void removeVariable_(VariablePtr);

    /** Split a jacobian matrix J into its components Ji corresponding to the
      * variables. 
      */
    void splitFullJacobian(const MatrixConstRef& J, bool keepProperties = false);

    // cache
    Eigen::VectorXd value_;
    std::map<Variable const*, MatrixWithProperties> jacobian_;

  private:
    int m_; //output size
    VariableVector variables_;
    std::map<Variable const*, bool> linear_;
  };


  inline const Eigen::VectorXd& FirstOrderProvider::value() const
  {
    return value_;
  }

  inline const MatrixWithProperties& FirstOrderProvider::jacobian(const Variable& x) const
  {
    return jacobian_.at(&x);
  }

  inline bool FirstOrderProvider::linearIn(const Variable& x) const
  {
    return linear_.at(&x);
  }

  inline int FirstOrderProvider::size() const
  {
    return m_;
  }

  inline const std::vector<VariablePtr>& FirstOrderProvider::variables() const
  {
    return variables_.variables();
  }

}  // namespace internal

}  // namespace tvm
