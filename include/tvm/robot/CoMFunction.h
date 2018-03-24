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

#include <tvm/Robot.h>

#include <tvm/function/abstract/Function.h>

#include <RBDyn/CoM.h>

namespace tvm
{

namespace robot
{

  /** This class implements a CoM function for a given robot */
  class TVM_DLLAPI CoMFunction : public function::abstract::Function
  {
  public:
    SET_UPDATES(CoMFunction, Value, Velocity, Jacobian, NormalAcceleration, JDot)

    /** Constructor
     *
     * Set the objective to the current CoM of robot
     *
     */
    CoMFunction(RobotPtr robot);

    /** Set the target CoM to the current robot's CoM */
    void reset();

    /** Get the current objective */
    inline const Eigen::Vector3d & com() const { return com_; }

    /** Set the objective */
    inline void com(const Eigen::Vector3d & com) { com_ = com; }
  protected:
    void updateValue();
    void updateVelocity();
    void updateJacobian();
    void updateNormalAcceleration();
    void updateJDot();

    RobotPtr robot_;

    /** Target */
    Eigen::Vector3d com_;

    /** CoM jacobian */
    rbd::CoMJacobian jac_;
  };

}

}
