/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Robot.h>
#include <tvm/geometry/Plane.h>

#include <tvm/function/abstract/Function.h>

#include <RBDyn/CoM.h>

namespace tvm
{

namespace robot
{

/** This function computes the distance of the CoM to a set of planes.
 *
 * By providing a consistent set of planes, the function can be used to keep
 * the CoM in a convex region of space.
 */
class TVM_DLLAPI CoMInConvexFunction : public function::abstract::Function
{
public:
  SET_UPDATES(CoMInConvexFunction, Value, Velocity, Jacobian, NormalAcceleration)

  /** Constructor
   *
   * By default, this function computes nothing
   *
   */
  CoMInConvexFunction(RobotPtr robot);

  /** Add a plane.
   *
   * This will add one dimension to the function output. This new value is
   * the distance to that plane.
   *
   * This function does not check whether this is consistent with planes that
   * were added previously.
   */
  void addPlane(geometry::PlanePtr plane);

  /** Remove all planes */
  void reset();

protected:
  void updateValue();
  void updateVelocity();
  void updateJacobian();
  void updateNormalAcceleration();

  RobotPtr robot_;

  /** Set of planes */
  std::vector<geometry::PlanePtr> planes_;

  /** CoM jacobian */
  rbd::CoMJacobian jac_;
  Eigen::Vector3d comSpeed_;
};

} // namespace robot

} // namespace tvm
