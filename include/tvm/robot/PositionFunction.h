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

  /** This class implements a position function for a given frame */
  class TVM_DLLAPI PositionFunction : public function::abstract::Function
  {
  public:
    SET_UPDATES(PositionFunction, Value, Velocity, Jacobian, NormalAcceleration)

    /** Constructor
     *
     * Set the objective to the current frame orientation
     *
     */
    PositionFunction(FramePtr frame);

    /** Set the target position to the current frame position */
    void reset();

    /** Get the current objective */
    inline const Eigen::Vector3d & position() const { return pos_; }

    /** Set the objective */
    inline void position(const Eigen::Vector3d & pos) { pos_ = pos; }
  protected:
    void updateValue();
    void updateVelocity();
    void updateJacobian();
    void updateNormalAcceleration();

    FramePtr frame_;

    /** Target */
    Eigen::Vector3d pos_;
  };

} // namespace robot

} // namespace tvm
