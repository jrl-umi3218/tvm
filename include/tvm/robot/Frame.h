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

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/function/abstract/Function.h>

#include <RBDyn/Jacobian.h>

namespace tvm
{

namespace robot
{

  /** A frame belonging to a robot
   *
   * Provides the frame position, jacobian, velocity and normal acceleration
   *
   */
  class TVM_DLLAPI Frame : public graph::abstract::Node<Frame>
  {
  public:
    SET_OUTPUTS(Frame, Position, Jacobian, Velocity, NormalAcceleration)
    SET_UPDATES(Frame, Position, Jacobian, Velocity, NormalAcceleration)

    /** Constructor
     *
     * Creates a frame belonging to a robot
     *
     * \param name Name of the frame
     *
     * \param robot Robot to which the frame is attached
     *
     * \param body Parent body of the frame
     *
     * \param X_b_f Static transformation from the body to the frame
     *
     */
    Frame(std::string name,
          RobotPtr robot,
          const std::string & body,
          sva::PTransformd X_b_f);

    /** Access the robot to which this frame belongs */
    inline const Robot & robot() const { return *robot_; }

    /** The frame's name */
    inline const std::string & name() const { return name_; }
    /** The frame's position in world coordinates */
    inline const sva::PTransformd & position() const { return position_; }
    /** The frame's jacobian in world coordinates */
    inline tvm::internal::MatrixWithProperties jacobian() const { return jacobian_; }
    /** The frame's velocity in world coordinates */
    inline sva::MotionVecd velocity() const { return velocity_; }
    /** The frame's normal acceleration in world coordinates */
    inline sva::MotionVecd normalAcceleration() const { return normalAcceleration_; }
  private:
    std::string name_;
    RobotPtr robot_;
    unsigned int bodyId_;
    rbd::Jacobian jac_;
    sva::PTransformd X_b_f_;

    /** Update functions and cache data */
    void updatePosition();
    sva::PTransformd position_;

    void updateJacobian();
    tvm::internal::MatrixWithProperties jacobian_;

    void updateVelocity();
    sva::MotionVecd velocity_;

    void updateNormalAcceleration();
    sva::MotionVecd normalAcceleration_;
  };

  using FramePtr = std::shared_ptr<Frame>;

}

}
