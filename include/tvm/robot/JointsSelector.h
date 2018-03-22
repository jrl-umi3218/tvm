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
