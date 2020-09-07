/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
