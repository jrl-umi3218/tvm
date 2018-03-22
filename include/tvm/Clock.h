#pragma once

/* Copyright 2017-2018 CNRS-UM LIRMM, CNRS-AIST JRL
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
#include <tvm/graph/abstract/Outputs.h>

#include <cstdint>

namespace tvm
{

  class ControlProblem;

  /** Represent a clock for the ControlProblem
   *
   * The current iteration of the problem can be accessed by time-dependant
   * data to trigger computations
   *
   * Outputs:
   *
   *   - Time is moved along as the iterations go on
   *
   */
  class TVM_DLLAPI Clock: public graph::abstract::Outputs
  {
    friend class ControlProblem;
  public:
    SET_OUTPUTS(Clock, Time)

    /** Constructor
     *
     * \param dt Timestep of the ControlProblem
     *
     */
    Clock(double dt);

    /** Returns the number of ticks elapsed since the start of the problem */
    inline uint64_t ticks() const { return ticks_; }

    /** Returns the timestep of the problem */
    inline double dt() const { return dt_; }

    /** Advance the clock by one tick */
    void advance();
  protected:
    Clock(const Clock &) = default;
    Clock & operator=(const Clock &) = default;
    Clock(Clock &&) = default;
    Clock & operator=(Clock &&) = default;
  private:
    const double dt_;
    /* Start ticks_ at -1 so that the first tick is 0 */
    uint64_t ticks_ = -1;
  };
}  // namespace tvm
