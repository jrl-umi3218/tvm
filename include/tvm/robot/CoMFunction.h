/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

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

} // namespace robot

} // namespace tvm
