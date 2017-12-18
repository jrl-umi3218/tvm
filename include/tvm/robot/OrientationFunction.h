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

#include <tvm/robot/Frame.h>
#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace robot
{

  /** This class implements an orientation function for a given frame */
  class TVM_DLLAPI OrientationFunction : public function::abstract::Function
  {
  public:
    SET_UPDATES(OrientationFunction, Value, Velocity, Jacobian, NormalAcceleration)

    /** Constructor
     *
     * Set the objective to the current frame orientation
     *
     */
    OrientationFunction(FramePtr frame);

    /** Set the target orientation to the current frame orientation */
    void reset();

    /** Get the current objective */
    inline const Eigen::Matrix3d & orientation() const { return ori_; }

    /** Set the objective */
    inline void orientation(const Eigen::Matrix3d & ori) { ori_ = ori; }
  protected:
    void updateValue();
    void updateVelocity();
    void updateJacobian();
    void updateNormalAcceleration();

    FramePtr frame_;

    /** Target */
    Eigen::Matrix3d ori_;
  };

} // namespace robot

} // namespace tvm
