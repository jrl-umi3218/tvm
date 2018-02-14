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

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/function/abstract/Function.h>
#include <tvm/robot/ConvexHull.h>

namespace tvm
{

namespace robot
{

  /** This class implements a collision function for two given objects */
  class TVM_DLLAPI CollisionFunction : public function::abstract::Function
  {
  public:
    SET_UPDATES(CollisionFunction, Value, Velocity, Jacobian, NormalAcceleration)

    /** Constructor
     *
     * \param dt Timestep
     *
     */
    CollisionFunction(double dt);

    /** Add a collision
     *
     * \param ch1 convex hull object for the first object
     *
     * \param ch2 convex hull object for the second object
     *
     */
    void addCollision(ConvexHullPtr ch1, ConvexHullPtr ch2);

    /** Remove all collisions */
    void reset();
  protected:
    /* Update functions */
    void updateValue();
    void updateVelocity();
    void updateJacobian();
    void updateNormalAcceleration();

    double dt_;

    struct CollisionData
    {
      struct ObjectData
      {
        Eigen::Vector3d nearestPoint_; // In body coordinates
        rbd::Jacobian jac_;
      };
      std::vector<ObjectData> objects_;
      ConvexHullPtr ch_[2];
      sch::CD_Pair pair_;
      Eigen::Vector3d normVecDist_;
      Eigen::Vector3d prevNormVecDist_ = Eigen::Vector3d::Zero();

      CollisionData();

      CollisionData(CollisionFunction & fn, ConvexHullPtr ch1, ConvexHullPtr ch2);
    };
    std::vector<CollisionData> colls_;

    /** Intermediate computation */
    Eigen::Vector3d closestPoints_[2];
    Eigen::MatrixXd fullJac_;
    Eigen::MatrixXd distJac_;
  };

} // namespace robot

} // namespace tvm
