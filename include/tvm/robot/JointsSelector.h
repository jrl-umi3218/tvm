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

#include <tvm/defs.h>
#include <tvm/Robot.h>

#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace robot
{

  /** This class implements joints' selection on a given fucntion */
  class TVM_DLLAPI JointsSelector : public function::abstract::Function
  {
  public:
    SET_UPDATES(JointsSelector, Jacobian, JDot)

    /** Construct a JointsSelector from a vector of active joints
     *
     * \param f Function selected by this JointsSelector
     *
     * \param robot Robot controlled by \p f
     *
     * \param activeJoints Joints active in this selector
     *
     * \throws If \p does not depend on \p robot or any joint in activeJoints
     * is not part of \p robot
     */
    static std::unique_ptr<JointsSelector> ActiveJoints(FunctionPtr f, RobotPtr robot, const std::vector<std::string> & activeJoints);

    /** Construct a JointsSelector from a vector of inactive joints
     *
     * \param f Function selected by this JointsSelector
     *
     * \param robot Robot controlled by \p f
     *
     * \param inactiveJoints Joints not active in this selector
     *
     * \throws If \p does not depend on \p robot or any joint in activeJoints
     * is not part of \p robot
     */
    static std::unique_ptr<JointsSelector> InactiveJoints(FunctionPtr f, RobotPtr robot, const std::vector<std::string> & inactiveJoints);

    const Eigen::VectorXd & value() const override { return f_->value(); }
    const Eigen::VectorXd & velocity() const override { return f_->velocity(); }
    const Eigen::VectorXd & normalAcceleration() const override { return f_->normalAcceleration(); }
  protected:
    /** Constructor
     *
     * \param f Function selected by this JointsSelector
     *
     * \param robot Robot controlled by \p f
     *
     * \param ffActive True if the selected joints include the free-flyer
     *
     * \param activeIndex For regular joints selected by this JointsSelector,
     * this iss a list of (start, size) pair corresponding to the list of
     * joints
     *
     */
    JointsSelector(FunctionPtr f, RobotPtr robot, bool ffActive, const std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex>> & activeIndex);
  protected:
    void updateJacobian();
    void updateJDot();

    FunctionPtr f_;
    RobotPtr robot_;
    bool ffActive_;
    std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex>> activeIndex_;
  };

} // namespace robot

} // namespace tvm
