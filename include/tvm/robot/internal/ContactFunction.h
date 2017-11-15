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


#include <tvm/function/abstract/Function.h>
#include <tvm/robot/Contact.h>

namespace tvm
{

namespace robot
{

namespace internal
{

  class ContactFunction : public function::abstract::Function
  {
  public:
    using Output = function::abstract::Function::Output;
    DISABLE_OUTPUTS(Output::JDot)
    SET_UPDATES(ContactFunction, Value, Velocity, NormalAcceleration, Jacobian)

    ContactFunction(ContactPtr contact,
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

    using GetJacobianFn = std::function<const tvm::internal::MatrixWithProperties&()>;
    std::map<const Variable*, GetJacobianFn> jacobian_getter_;
  };

}

}

}
