/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

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
 * Provides the frame position, jacobian, velocity and normal acceleration.
 * These signals are correctly initialized on the object's creation.
 *
 * Outputs:
 * - Position: position of the frame in world coordinates
 * - Jacobian: jacobian of the frame in world coordinates
 * - Velocity: velocity of the frame in world coordinates
 * - NormalAcceleration: normal acceleration of the frame in world
 *   coordinates
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
  Frame(std::string name, RobotPtr robot, const std::string & body, sva::PTransformd X_b_f);

  /** Access the robot to which this frame belongs (const) */
  inline const Robot & robot() const { return *robot_; }

  /** Access the robot to which this frame belongs */
  inline Robot & robot() { return *robot_; }

  /** Access the internal Jacobian object to perform extra-computation */
  inline rbd::Jacobian & rbdJacobian() { return jac_; }

  /** Access the internal Jacobian object to perform extra-computation (const) */
  inline const rbd::Jacobian & rbdJacobian() const { return jac_; }

  /** The frame's name */
  inline const std::string & name() const { return name_; }
  /** The frame's position in world coordinates */
  inline const sva::PTransformd & position() const { return position_; }
  /** The frame's jacobian in world coordinates */
  inline const tvm::internal::MatrixWithProperties & jacobian() const { return jacobian_; }
  /** The frame's velocity in world coordinates */
  inline const sva::MotionVecd & velocity() const { return velocity_; }
  /** The frame's normal acceleration in world coordinates */
  inline const sva::MotionVecd & normalAcceleration() const { return normalAcceleration_; }
  /** Returns the frame's parent body */
  const std::string & body() const;

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
  Eigen::MatrixXd jacTmp_;
  tvm::internal::MatrixWithProperties jacobian_;

  void updateVelocity();
  sva::MotionVecd velocity_;

  void updateNormalAcceleration();
  sva::MotionVecd normalAcceleration_;
};

using FramePtr = std::shared_ptr<Frame>;

} // namespace robot

} // namespace tvm
