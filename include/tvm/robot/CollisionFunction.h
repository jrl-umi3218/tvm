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
    SET_UPDATES(CollisionFunction, Value, Velocity, Jacobian, Time, NormalAcceleration)

    /** Constructor
     *
     * \param dt Timestep
     *
     */
    CollisionFunction(Clock & clock);

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
    void updateTimeDependency();
    void updateNormalAcceleration();

    Clock & clock_;
    uint64_t last_tick_ = 0;

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
      Eigen::Vector3d speedVec_ = Eigen::Vector3d::Zero();

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
