/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/internal/FirstOrderProvider.h>
#include <tvm/utils/internal/map.h>

#include <Eigen/Core>

#include <map>

namespace tvm
{

namespace function
{

namespace abstract
{

/** Base class defining the classical outputs for a function
 *
 * \dot
 * digraph "update graph" {
 *   rankdir="LR";
 *   {
 *     rank = same; node [shape=hexagon];
 *     Value; Jacobian; Velocity;
 *     NormalAcceleration; JDot;
 *   }
 *   {
 *     rank = same; node [style=invis, label=""];
 *     outValue; outJacobian; outVelocity;
 *     outNormalAcceleration; outJDot;
 *   }
 *   Value -> outValue [label="value()"];
 *   Jacobian -> outJacobian [label="jacobian(x_i)"];
 *   Velocity -> outVelocity [label="velocity()"];
 *   NormalAcceleration -> outNormalAcceleration [label="normalAcceleration()"];
 *   JDot -> outJDot [label="JDot(x_i)"];
 * }
 * \enddot
 */
class TVM_DLLAPI Function : public tvm::internal::FirstOrderProvider
{
public:
  SET_OUTPUTS(Function, Velocity, NormalAcceleration, JDot)

  /** Note: by default, these methods return the cached value.
   * However, they are virtual in case the user might want to bypass the cache.
   * This would be typically the case if he/she wants to directly return the
   * output of another method, e.g. return the jacobian of an other Function.
   */
  virtual const Eigen::VectorXd & velocity() const;
  virtual const Eigen::VectorXd & normalAcceleration() const;
  virtual MatrixConstRef JDot(const Variable & x) const;

protected:
  struct slice_jdot
  {
    using Type = MatrixRef;
    using ConstType = MatrixConstRef;
    static Type get(Eigen::MatrixXd & M, const Range & r) { return M.middleCols(r.start, r.dim); }
    static ConstType get(const Eigen::MatrixXd & M, const Range & r) { return M.middleCols(r.start, r.dim); }
  };
  /** Constructor for a function with value in \f$ \mathbb{R}^m \f$.
   *
   * \param m the size of the function/constraint image space, i.e. the row
   * size of the jacobians (or equivalently in this case the size of the
   * output value).
   */
  Function(int m = 0);

  /** Constructor for a function with value in a specified space.
   *
   * \param image Description of the image space
   */
  Function(Space image);

  /** Resize all cache members corresponding to active output*/
  void resizeCache() override;
  void resizeVelocityCache();
  void resizeNormalAccelerationCache();
  void resizeJDotCache();

  void addVariable_(VariablePtr v) override;
  void removeVariable_(VariablePtr v) override;

  // cache
  Eigen::VectorXd velocity_;
  Eigen::VectorXd normalAcceleration_;
  utils::internal::MapWithVariableAsKey<Eigen::MatrixXd, slice_jdot> JDot_;

private:
  // we retain the variables' derivatives shared_ptr to ensure the reference is never lost
  std::vector<VariablePtr> variablesDot_;
};

inline const Eigen::VectorXd & Function::velocity() const { return velocity_; }

inline const Eigen::VectorXd & Function::normalAcceleration() const { return normalAcceleration_; }

inline MatrixConstRef Function::JDot(const Variable & x) const
{
  return JDot_.at(&x, tvm::utils::internal::with_sub{});
}

} // namespace abstract

} // namespace function

} // namespace tvm
