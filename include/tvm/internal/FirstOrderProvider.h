/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include "tvm/internal/CallbackManager.h"
#include <iostream>
#include <tvm/defs.h>

#include <tvm/Variable.h>
#include <tvm/VariableVector.h>
#include <tvm/graph/abstract/Node.h>
#include <tvm/internal/MatrixWithProperties.h>
#include <tvm/utils/internal/MapWithVariableAsKey.h>

#include <Eigen/Core>

#include <algorithm>
#include <map>
#include <vector>

namespace tvm::internal
{

// A callback class to be fired when variables are added or removed.
// This is to ensure the function variables are correctly synced to the tasks of the problem when the
// functions add or remove variables themselves (for example with contacts).
class UpdateVariableCallback : public internal::CallbackManager
{
public:
  UpdateVariableCallback() : internal::CallbackManager() {}

  void variableUpdated(VariablePtr v)
  {
    // std::cout << "UpdateVariableCallback: Calling callback for variable: " << v->name() << std::endl;
    internal::CallbackManager::run();
  }
};

/** Describes an entity that can provide a value and its jacobian
 *
 * \dot
 * digraph "update graph" {
 *   rankdir="LR";
 *   {
 *     rank = same;
 *     node [shape=hexagon];
 *     Value; Jacobian;
 *   }
 *   {
 *     rank = same;
 *     node [style=invis, label=""];
 *     outValue; outJacobian;
 *   }
 *   Value -> outValue [label="value()"];
 *   Jacobian -> outJacobian [label="jacobian(x_i)"];
 * }
 * \enddot
 */
class TVM_DLLAPI FirstOrderProvider : public graph::abstract::Node<FirstOrderProvider>
{
public:
  SET_OUTPUTS(FirstOrderProvider, Value, Jacobian)

  /** \internal by default, these methods return the cached value.
   * However, they are virtual in case the user might want to bypass the cache.
   * This would be typically the case if he/she wants to directly return the
   * output of another method, e.g. return the jacobian of an other Function.
   */
  /** Return the value of this entity*/
  virtual const Eigen::VectorXd & value() const;
  /** Return the jacobian matrix of this entity corresponding to the variable
   * \p x
   */
  virtual MatrixConstRefWithProperties jacobian(const Variable & x) const;

  /** Linearity w.r.t \p x*/
  bool linearIn(const Variable & x) const;

  const Space & imageSpace() const;

  /** Return the image space size.*/
  int size() const;
  /** Size of the output value (representation size of the image space).*/
  int rSize() const;
  /** Size of the tangent space to the image space (or equivalently row size of the jacobian matrices).*/
  int tSize() const;

  /** Return the variables*/
  const VariableVector & variables() const;

  UpdateVariableCallback & updateVariableCallback() noexcept { return updateVariableCallback_; }

protected:
  struct slice_linear
  {
    using Type = bool;
    using ConstType = bool;
    static bool get(bool & b, const Range &) { return b; }
    static bool get(const bool & b, const Range &) { return b; }
  };

  struct slice_matrix
  {
    using Type = tvm::internal::ObjectWithProperties<MatrixRef, true>;
    using ConstType = MatrixConstRefWithProperties;
    static MatrixProperties slice(const MatrixProperties & p)
    {
      if(p.isZero())
        return {MatrixProperties::ZERO, MatrixProperties::Constness(p.isConstant())};
      else
        return {MatrixProperties::Constness(p.isConstant())};
    }
    static Type get(MatrixWithProperties & M, const Range & r)
    {
      return {M.middleCols(r.start, r.dim), M.properties()};
    }
    static ConstType get(const MatrixWithProperties & M, const Range & r)
    {
      return {M.middleCols(r.start, r.dim), slice(M.properties())};
    }
  };

  /** Constructor for a function/constraint with value in \f$ \mathbb{R}^m \f$.
   *
   * \param m the size of the function/constraint image space, i.e. the row
   * size of the jacobians (or equivalently in this case the size of the
   * output value).
   */
  FirstOrderProvider(int m);

  /** Constructor for a function/constraint with value in a specified space.
   *
   * \param image Description of the image space
   */
  FirstOrderProvider(Space image);

  /** Resize all cache members corresponding to active outputs.
   *
   * This can be overridden in case you do not need all of the default
   * mechanism (typically if you will not use part of the cache).
   * If you override to perform additional operations, do not forget to
   * call this base version in the derived classes.
   */
  virtual void resizeCache();

  /** Sub-methods of resizeCache resizing the value cache vector (if used).
   * To be used by derived classes that need this level of granularity.
   */
  void resizeValueCache();
  /** Sub-methods of resizeCache resizing the jacobian cache matrices (if
   * used).
   * To be used by derived classes that need this level of granularity.
   */
  void resizeJacobianCache();

  /** Add a variable. Cache is automatically updated.
   * \param v The variable to add
   * \param linear Specify that the entity is depending linearly on the
   * variable or not.
   */
  void addVariable(VariablePtr v, bool linear);
  /** Remove variable \p v. Cache is automatically updated. */
  void removeVariable(VariablePtr v);

  /** Add a variable vector.
   *
   * Convenience function similar to adding each elements of the vector individually with the same linear parameter
   *
   * \see addVariable(VariablePtr, bool)
   *
   */
  void addVariable(const VariableVector & v, bool linear);

  /** To be overridden by derived classes that need to react to
   * the addition of a variable. Called at the end of addVariable();
   */
  virtual void addVariable_(VariablePtr);

  /** To be overridden by derived classes that need to react to
   * the removal of a variable. Called at the end of removeVariable();
   */
  virtual void removeVariable_(VariablePtr);

  /** Split a jacobian matrix J into its components Ji corresponding to the
   * provided variables.
   *
   * \param J The matrix to be split
   * \param vars The vector of variables giving the layout of J. It is the
   * user's responsibility to ensure these variables are part of variables_
   * and that J has the correct size.
   * \param keepProperties If true, the properties associated with matrices
   * Ji are kept, if not they are reset to default.
   */
  void splitJacobian(const MatrixConstRef & J, const std::vector<VariablePtr> & vars, bool keepProperties = false);

  /** Overload for VariableVector operations */
  inline void splitJacobian(const MatrixConstRef & J, const VariableVector & vars, bool keepProperties = false)
  {
    splitJacobian(J, vars.variables(), keepProperties);
  }

  // cache
  Eigen::VectorXd value_;
  utils::internal::MapWithVariableAsKey<MatrixWithProperties, slice_matrix, true> jacobian_;

protected:
  /** Resize the function */
  void resize(int m);

  Space imageSpace_; // output space
  VariableVector variables_;
  utils::internal::MapWithVariableAsKey<bool, slice_linear> linear_;
  UpdateVariableCallback updateVariableCallback_;
};

inline const Eigen::VectorXd & FirstOrderProvider::value() const { return value_; }

inline MatrixConstRefWithProperties FirstOrderProvider::jacobian(const Variable & x) const
{
  return jacobian_.at(&x, tvm::utils::internal::with_sub{});
}

inline bool FirstOrderProvider::linearIn(const Variable & x) const
{
  return linear_.at(&x, tvm::utils::internal::with_sub{});
}

inline const Space & FirstOrderProvider::imageSpace() const { return imageSpace_; }

inline int FirstOrderProvider::size() const { return imageSpace_.size(); }

inline int FirstOrderProvider::rSize() const { return imageSpace_.rSize(); }
inline int FirstOrderProvider::tSize() const { return imageSpace_.tSize(); }

inline const VariableVector & FirstOrderProvider::variables() const { return variables_; }

} // namespace tvm::internal
