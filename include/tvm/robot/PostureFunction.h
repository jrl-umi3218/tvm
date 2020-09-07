/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

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

} // namespace robot

} // namespace tvm
