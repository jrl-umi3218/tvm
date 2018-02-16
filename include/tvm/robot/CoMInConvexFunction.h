#pragma once

/* Copyright 2017-2018 CNRS-UM LIRMM, CNRS-AIST JRL
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
