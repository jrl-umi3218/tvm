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

#include <tvm/Clock.h>
#include <tvm/Variable.h>
#include <tvm/VariableVector.h>
#include <tvm/graph/abstract/Node.h>

#include <RBDyn/FD.h>
#include <RBDyn/MultiBody.h>
#include <RBDyn/MultiBodyConfig.h>
#include <RBDyn/MultiBodyGraph.h>

namespace tvm
{

  /** Represent a Robot
   *
   * A robot is constructed by providing instances of MultiBodyGraph, MultiBody
   * and MultiBodyConfig that are created from each other. It provides signals
   * that are relevant for computing quantities related to a robot.
   *
   * Variables:
   * - q (split between free-flyer and joints)
   * - tau (see Outputs)
   *
   * Outputs:
   *
   * - Kinematics: kinematics quantity (computed by RBDyn::FK)
   * - Dynamics: dynamics quantity (computed by RBDyn::FV, depends on
   *   Kinematics)
   * - Acceleration: compute acceleration quantities (computed by RBDyn::FA,
   *   depends on Dynamics)
   * - tau: generalized torque vector, this output isn't currently linked to
   *   any computation
   * - CoM: center of mass signal
   * - H: inertia matrix signal
   * - C: non-linear effect vector signal (Coriolis, gravity, external forces)
   *
   */
  class TVM_DLLAPI Robot : public graph::abstract::Node<Robot>
  {
  public:
    SET_OUTPUTS(Robot, Kinematics, Dynamics, Acceleration, tau, CoM, H, C)
    SET_UPDATES(Robot, Time, Kinematics, Dynamics, Acceleration, CoM, H, C)

    /** Constructor
     *
     * \param clock Clock used in the ControlProblem
     *
     * \param name Name of the robot
     *
     * \param mbg MultiBodyGraph used to create mb/mbc
     *
     * \param mb MultiBody representing the robot's kinematic structure
     *
     * \param mbc MultiBodyConfig giving the robot's initial configuration
     *
     */
    Robot(Clock & clock,
          const std::string & name, rbd::MultiBodyGraph & mbg, rbd::MultiBody
          mb, rbd::MultiBodyConfig mbc);


    /** Access the robot's name */
    inline const std::string & name() const { return name_; }

    /** Access q variable (const) */
    inline const VariableVector & q() const { return q_; }
    /** Access q variable */
    inline VariableVector & q() { return q_; }

    /** Access free-flyer variable (const) */
    inline const VariablePtr & qFreeFlyer() const { return q_ff_; }
    /** Access free-flyer variable */
    inline VariablePtr & qFreeFlyer() { return q_ff_; }

    /** Access joints variable (const) */
    inline const VariablePtr & qJoints() const { return q_joints_; }
    /** Access joints variable */
    inline VariablePtr & qJoints() { return q_joints_; }

    /** Access tau variable (const) */
    inline const VariablePtr & tau() const { return tau_; }
    /** Access tau variable */
    inline VariablePtr & tau() { return tau_; }

    /** Access the robot's related rbd::MultiBody (const) */
    inline const rbd::MultiBody & mb() const { return mb_; }
    /** Access the robot's related rbd::MultiBody */
    inline rbd::MultiBody & mb() { return mb_; }

    /** Access the robot's related rbd::MultiBodyConfig (const) */
    inline const rbd::MultiBodyConfig & mbc() const { return mbc_; }
    /** Access the robot's related rbd::MultiBodyConfig */
    inline rbd::MultiBodyConfig & mbc() { return mbc_; }

    /** Access the vector of normal acceleration expressed in the body's frame (const) */
    inline const std::vector<sva::MotionVecd> & normalAccB() const { return normalAccB_; }
    /** Access the vector of normal acceleration expressed in the body's frame */
    inline std::vector<sva::MotionVecd> & normalAccB() { return normalAccB_; }

    /** Access the inertia matrix */
    inline const Eigen::MatrixXd & H() const { return fd_.H(); }
    /** Access the non-linear effect vector */
    inline const Eigen::VectorXd & C() const { return fd_.C(); }

    /** Access the CoM position */
    inline const Eigen::Vector3d & com() const { return com_; }

    /** Access the transformation that allows to retrieve the original base of a body */
    inline const sva::PTransformd & bodyTransform(const std::string & b) const { return bodyTransforms_.at(b); }
  private:
    Clock & clock_;
    uint64_t last_tick_ = 0;
    std::string name_;
    rbd::MultiBody mb_;
    rbd::MultiBodyConfig mbc_;
    std::vector<sva::MotionVecd> normalAccB_;
    rbd::ForwardDynamics fd_;
    std::map<std::string, sva::PTransformd> bodyTransforms_;
    VariablePtr q_ff_;
    VariablePtr q_joints_;
    VariableVector q_;
    VariableVector dq_;
    VariableVector ddq_;
    VariablePtr tau_;
    Eigen::Vector3d com_;
  private:
    void computeNormalAccB();

    /** Update the Robot's variables based on the output of the solver
     *
     * It will:
     * 1. Put dot(q,2) into mbc.alphaD
     * 2. Run rbd::eulerIntegration with dt
     * 3. Output mbc.alpha into dot(q)
     * 4. Output mbc.q into q
     *
     * \param dt Integration timestep
     *
     */
    void updateTimeDependency();
    void updateKinematics();
    void updateDynamics();
    void updateAcceleration();
    void updateH();
    void updateC();
    void updateCoM();
  };

}
