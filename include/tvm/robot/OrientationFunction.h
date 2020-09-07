/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/function/abstract/Function.h>
#include <tvm/robot/Frame.h>

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
