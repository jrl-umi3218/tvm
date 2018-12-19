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


#include <tvm/api.h>
#include <tvm/Robot.h>

#include <tvm/function/abstract/LinearFunction.h>

#include <tvm/robot/enums.h>
#include <tvm/robot/internal/GeometricContactFunction.h>
#include <tvm/robot/internal/FrictionCone.h>

#include <RBDyn/FD.h>

namespace tvm
{

class ControlProblem;

namespace robot
{

namespace internal
{

  /** Implement the equation of motion for a given robot.
   *
   * It can be given contacts that will be integrated into the equation of
   * motion (\see DynamicFunction::addContact) or contacts that will be removed
   * from the equation (\see DynamicFunction::removeContact).
   *
   * It manages the force variables related to these contacts.
   *
   * It also takes care of linearizing the contacts, either by having them in a
   * linearized form in the solver (lambdas) or as forces. The related
   * constraint (mu*f_n >= ||f_t||) can be added to the problem through this
   * function.
   *
   * Notably, it does not take care of enforcing Newton 3rd law of motion when
   * two actuated robots are in contact.
   *
   */
  class TVM_DLLAPI DynamicFunction : public function::abstract::LinearFunction
  {
  public:
    using Output = function::abstract::LinearFunction::Output;
    DISABLE_OUTPUTS(Output::JDot)
    SET_UPDATES(DynamicFunction, Jacobian, B)

    /** Construct the equation of motion for a given robot */
    DynamicFunction(RobotPtr robot);

    /** Add a contact to the function
     *
     * This adds forces variables for every contact point belonging to the
     * robot of this dynamic function.
     *
     * \param contact Contact that will be added
     *
     * \param linearize If true, linearize the friction cone using generatrices
     *
     * \param mu Friction coefficient
     *
     * \param nrGen Number of generatrices for the cone (only applicable when
     * linearize is true)
     *
     * Returns true if a contact has been added */
    bool addContact(ContactPtr contact,
                    bool linearize, double mu, unsigned int nrGen);

    /** Remove a contact
     *
     * This has no effect if this contact was not added before
     *
     */
    void removeContact(const Contact::Id & id);

    /** Return the contact force at the contact.
     *
     * The force is null if this contact has not been added or does not
     * concern the related robot.
     */
    sva::ForceVecd contactForce(const Contact::Id & id) const;

    /** Add constraints to the given problem
     *
     * The constraints express that the contact should not slip, i.e. mu*f_n >
     * ||f_t||, where mu was set when the contact was added.
     *
     */
    void addPositiveLambdaToProblem(ControlProblem & problem);
  protected:
    void updateb();

    RobotPtr robot_;

    /** Holds data for the force part of the motion equation */
    struct ForceContact
    {
      /** Contact id */
      Contact::Id id_;

      /** True if the contact has been linearized */
      bool linearized_;

      /** Force (1 3d variable per contact point)
       *
       * OR
       *
       * lambdas (1 nrGen-d variable per contact point)
       *
       */
      std::vector<tvm::VariablePtr> forces_;

      /** Compute the resulting force on the contact */
      std::function<sva::ForceVecd(const ForceContact&)> force_;

      /** Used for intermediate Jacobian computation */
      Eigen::MatrixXd force_jac_;
      Eigen::MatrixXd full_jac_;

      /** Update jacobians */
      std::function<void(ForceContact&, DynamicFunction&)> updateJacobians_;
    };
    std::vector<ForceContact> contacts_;

    void updateJacobian();

    void addContact_(const Contact::View & contact, bool linearize,
                     double mu, unsigned int nrGen, double dir);

    std::vector<ForceContact>::iterator getContact(const Contact::Id & id);
    std::vector<ForceContact>::const_iterator getContact(const Contact::Id & id) const;
  };

}

}

}
