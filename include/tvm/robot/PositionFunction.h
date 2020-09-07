/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/function/abstract/Function.h>
#include <tvm/robot/Frame.h>

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
