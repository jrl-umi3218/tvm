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

#include <tvm/Variable.h>
#include <tvm/graph/abstract/Node.h>

#include <RBDyn/FD.h>
#include <RBDyn/MultiBody.h>
#include <RBDyn/MultiBodyConfig.h>

namespace tvm
{

  /** Represent a Robot */
  class TVM_DLLAPI Robot : public graph::abstract::Node<Robot>
  {
  public:
    SET_OUTPUTS(Robot, q, tau)

    /** Constructor
     *
     * \param name Name of the robot
     *
     * \param mb MultiBody representing the robot's kinematic structure
     *
     * \param mbc MultiBodyConfig giving the robot's initial configuration
     *
     */
    Robot(const std::string & name,
          rbd::MultiBody mb, rbd::MultiBodyConfig mbc);

    /** Update the Robot's complete state */
    void updateTimeDependency(double dt);

    inline const std::string & name() const { return name_; }

    inline const VariablePtr & q() const { return q_; }
    inline VariablePtr & q() { return q_; }

    inline const VariablePtr & tau() const { return tau_; }
    inline VariablePtr & tau() { return tau_; }

    inline const rbd::MultiBody & mb() const { return mb_; }
    inline rbd::MultiBody & mb() { return mb_; }

    inline const rbd::MultiBodyConfig & mbc() const { return mbc_; }
    inline rbd::MultiBodyConfig & mbc() { return mbc_; }

    inline const std::vector<sva::MotionVecd> & normalAccB() const { return normalAccB_; }
    inline std::vector<sva::MotionVecd> & normalAccB() { return normalAccB_; }

  private:
    std::string name_;
    rbd::MultiBody mb_;
    rbd::MultiBodyConfig mbc_;
    std::vector<sva::MotionVecd> normalAccB_;
    rbd::ForwardDynamics fd_;
    VariablePtr q_;
    VariablePtr tau_;
  private:
    void computeNormalAccB();
  };

}
