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

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm
{

namespace task_dynamics
{

  /** Compute \dot{e}* = -kp*f (Kinematic order)
   *
   * FIXME have a version with diagonal or sdp gain matrix
   */
  class TVM_DLLAPI Proportional: public abstract::TaskDynamics
  {
  public:
    class Impl: public abstract::TaskDynamicsImpl
    {
    public:
      Impl(FunctionPtr f, double kp);
      void updateValue() override;

    private:
      double kp_;
    };

    Proportional(double kp);

  protected:
    std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f) const override;

  private:
    double kp_;
  };

  /** Alias for convenience */
  using P = Proportional;
}  // namespace task_dynamics

}  // namespace tvm
