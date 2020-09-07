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
