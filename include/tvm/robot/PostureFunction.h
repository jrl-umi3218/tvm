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

namespace tvm
{

namespace robot
{

  /** This class implements a posture function for a given robot */
  class TVM_DLLAPI PostureFunction : public function::abstract::Function
  {
  public:
    SET_UPDATES(PostureFunction, Value, Velocity)

    /** Constructor
     *
     * Set the objective to the current posture of robot
     *
     */
    PostureFunction(RobotPtr robot);

    /** Set the target posture to the current robot's posture */
    void reset();

    /** Set the target for a given joint
     *
     *  \param j Joint name
     *
     *  \param q Target configuration
     *
     */
    void posture(const std::string & j, const std::vector<double> & q);

    /** Set the fully body posture */
    void posture(const std::vector<std::vector<double>> & p);
  protected:
    void updateValue();

    void updateVelocity();

    RobotPtr robot_;

    /** Target */
    std::vector<std::vector<double>> posture_;

    /** Starting joint */
    int j0_;
  };

}

}
