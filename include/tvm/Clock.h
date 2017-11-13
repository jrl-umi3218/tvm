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

#include <tvm/graph/abstract/Outputs.h>

namespace tvm
{
  /* Rationale: we have a unique timer allowed in a ControlProblem (more
   * would be possible easily but what for ?).
   *
   * When updating the problem, the following order is applied:
   *
   * 1. increment the clock
   *
   * 2. call all updateTimeDependency
   *
   * 3. call the update plan
   *
   */
  class Clock: public graph::abstract::Outputs
  {
  public:
    SET_OUTPUTS(Clock, CurrentTime)

    Clock(double initTime=0);

    void increment(double dt);
    void reset(double resetTime = 0);
    double currentTime() const;

  private:
    double t_;
  };
}  // namespace tvm
