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
class TVM_DLLAPI Clock : public graph::abstract::Outputs
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
  uint64_t ticks_ = 0;
};
} // namespace tvm
