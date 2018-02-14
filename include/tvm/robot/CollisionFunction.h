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
#include <tvm/robot/Frame.h>

#include <sch/CD/CD_Pair.h>

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
     * \param f1 Frame of the first object
     *
     * \param o1 sch object representing the first object
     *
     * \param X_f1_o1 \p o1 position will be set at each iteration to \f$ {}^{o1}X_{f1} f1.position() \f$
     *
     * \param f2 Frame of the second object
     *
     * \param o2 sch object representing the second object
     *
     * \param X_f2_o2 \p o2 position will be set at each iteration to \f$ {}^{o2}X_{f2} f2.position() \f$
     *
     */
    void addCollision(
      FramePtr f1, std::shared_ptr<sch::S_Object> o1, const sva::PTransformd & X_f1_o1,
      FramePtr f2, std::shared_ptr<sch::S_Object> o2, const sva::PTransformd & X_f2_o2);

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
        FramePtr f_;
        sch::S_Object * o_;
        sva::PTransformd X_f_o_;
        Eigen::Vector3d nearestPoint_; // In body coordinates
        rbd::Jacobian jac_;
      };
      std::vector<ObjectData> objects_;
      std::shared_ptr<sch::S_Object> o1_;
      std::shared_ptr<sch::S_Object> o2_;
      sch::CD_Pair pair_;
      Eigen::Vector3d normVecDist_;
      Eigen::Vector3d prevNormVecDist_ = Eigen::Vector3d::Zero();

      CollisionData();

      CollisionData(CollisionFunction & fn,
        FramePtr f1, std::shared_ptr<sch::S_Object> o1, const sva::PTransformd & X_f1_o1,
        FramePtr f2, std::shared_ptr<sch::S_Object> o2, const sva::PTransformd & X_f2_o2);
    };
    std::vector<CollisionData> colls_;

    /** Intermediate computation */
    Eigen::Vector3d closestPoints_[2];
    Eigen::MatrixXd fullJac_;
    Eigen::MatrixXd distJac_;
  };

} // namespace robot

} // namespace tvm
