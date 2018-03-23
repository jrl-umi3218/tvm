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

#include <tvm/api.h>
#include <tvm/function/abstract/Function.h>
#include <tvm/robot/Contact.h>

namespace tvm
{

namespace robot
{

namespace internal
{

  /** Represents a geometric contact function
   *
   * Given a Contact (f1, f2) belonging to (r1, r2) this is the difference
   * between their current relative position and their initial relative
   * position. This is a function of r1.q and r2.q if r1 or r2 is not actuated,
   * then it is not taken into account in computations. If both are not
   * actuated this does nothing.
   *
   * The degrees of freedom computed by this function can be specified through
   * a 6x6 dof matrix.
   *
   * Outputs:
   *
   * - Value: dof*transformError(X_f1_f2, X_f1_f2_init)
   * - Velocity: dof*(f1.velocity - f2.velocity)
   * - NormalAcceleration: dof*(f1.NA - f2.NA)
   * - Jacobian: defined for r1.q as dof*f1.J and for r2.q as -dof*f2.J
   *
   */
  class TVM_DLLAPI GeometricContactFunction : public function::abstract::Function
  {
  public:
    using Output = function::abstract::Function::Output;
    DISABLE_OUTPUTS(Output::JDot)
    SET_UPDATES(GeometricContactFunction, Value, Velocity, NormalAcceleration, Jacobian)

    /** Constructor
     *
     * \param contact The Contact that this function computes
     *
     * \param dof The function dof matrix
     *
     */
    GeometricContactFunction(ContactPtr contact,
                             Eigen::Matrix6d dof);
  private:
    ContactPtr contact_;
    Eigen::Matrix6d dof_;
    bool has_f1_;
    bool has_f2_;

    bool first_update_ = true;
    sva::PTransformd X_f1_f2_init_;

    void updateValue();
    void updateVelocity();
    void updateNormalAcceleration();
    void updateJacobian();
  };

}

}

}
